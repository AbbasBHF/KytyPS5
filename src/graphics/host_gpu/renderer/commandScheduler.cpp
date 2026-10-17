#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>

#if defined(TRACY_ENABLE)
#  define TRACY_VK_USE_SYMBOL_TABLE
#  include <tracy/TracyVulkan.hpp>
#endif

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4, uint64_t shader_hash) {
	LOGE("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 " shader_hash=0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4, shader_hash);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 " shader_hash=0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4, shader_hash);
	std::fflush(stdout);
}

void ReportDeviceFault(GraphicContext& graphics, uint64_t tick) {
	if (!graphics.device_fault_enabled) {
		LOGE("VK_EXT_device_fault diagnostics unavailable (extension/feature not enabled)\n");
		return;
	}

	const auto get_fault_info = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT;
	if (get_fault_info == nullptr) {
		LOGE("VK_EXT_device_fault is enabled, but vkGetDeviceFaultInfoEXT was not loaded\n");
		return;
	}

	vk::DeviceFaultCountsEXT counts {};
	auto fault_result = static_cast<vk::Result>(
	    get_fault_info(static_cast<VkDevice>(graphics.device),
	                    reinterpret_cast<VkDeviceFaultCountsEXT*>(&counts), nullptr));
	if (fault_result != vk::Result::eSuccess) {
		LOGE("vkGetDeviceFaultInfoEXT count query failed: %s (%d)\n",
		     vk::to_string(fault_result).c_str(), static_cast<int>(fault_result));
		return;
	}

	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
	constexpr uint64_t MaxVendorBinaryBytes = 64ull * 1024ull * 1024ull;
	std::vector<uint8_t> vendor_binary;
	if (graphics.device_fault_vendor_binary_enabled && counts.vendorBinarySize > 0 &&
	    counts.vendorBinarySize <= MaxVendorBinaryBytes) {
		vendor_binary.resize(static_cast<size_t>(counts.vendorBinarySize));
	} else if (counts.vendorBinarySize > MaxVendorBinaryBytes) {
		LOGE("Device fault vendor binary is too large to capture: %" PRIu64 " bytes\n",
		     static_cast<uint64_t>(counts.vendorBinarySize));
	}
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	info.pVendorBinaryData = vendor_binary.empty() ? nullptr : vendor_binary.data();
	fault_result = static_cast<vk::Result>(
	    get_fault_info(static_cast<VkDevice>(graphics.device),
	                    reinterpret_cast<VkDeviceFaultCountsEXT*>(&counts),
	                    reinterpret_cast<VkDeviceFaultInfoEXT*>(&info)));
	if (fault_result != vk::Result::eSuccess) {
		LOGE("vkGetDeviceFaultInfoEXT data query failed: %s (%d)\n",
		     vk::to_string(fault_result).c_str(), static_cast<int>(fault_result));
		return;
	}

	LOGE("Device fault report: description='%.*s', addresses=%u, vendor_records=%u, "
	     "vendor_binary_bytes=%" PRIu64 "\n",
	     VK_MAX_DESCRIPTION_SIZE, info.description.data(), counts.addressInfoCount,
	     counts.vendorInfoCount,
	     static_cast<uint64_t>(counts.vendorBinarySize));
	for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
		const auto& address = addresses[i];
		LOGE("  fault address[%u]: type=%s address=0x%016" PRIx64 " precision=%" PRIu64 "\n",
		     i, vk::to_string(address.addressType).c_str(),
		     static_cast<uint64_t>(address.reportedAddress),
		     static_cast<uint64_t>(address.addressPrecision));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
		const auto& vendor = vendors[i];
		LOGE("  vendor fault[%u]: description='%.*s' code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n",
		     i, VK_MAX_DESCRIPTION_SIZE, vendor.description.data(), vendor.vendorFaultCode,
		     vendor.vendorFaultData);
	}
	if (!vendor_binary.empty()) {
		std::error_code ec;
		std::filesystem::create_directories("_Shaders", ec);
		char path[96] {};
		std::snprintf(path, sizeof(path), "_Shaders/device_fault_%" PRIu64 ".bin", tick);
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		if (file) {
			file.write(reinterpret_cast<const char*>(vendor_binary.data()),
			           static_cast<std::streamsize>(vendor_binary.size()));
			if (file) {
				LOGE("Saved device fault vendor binary: %s (%zu bytes)\n", path,
				     vendor_binary.size());
			} else {
				LOGE("Failed writing device fault vendor binary: %s\n", path);
			}
		} else {
			LOGE("Could not open device fault vendor binary output: %s\n", path);
		}
	}
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
#if defined(TRACY_ENABLE)
	if (m_tracy_gpu_context != nullptr) {
		TracyVkDestroy(static_cast<TracyVkCtx>(m_tracy_gpu_context));
		m_tracy_gpu_context = nullptr;
	}
#endif
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::LogRecentSubmissionsForReadback(uint64_t target_tick) const {
	const auto count = std::min<size_t>(m_recent_submission_count, 8);
	LOGD("Recent queue submissions before readback wait: target_tick=%" PRIu64 " count=%zu\n",
	     target_tick, count);
	for (size_t i = 0; i < count; ++i) {
		const auto index = (m_recent_submission_next + RecentSubmissionCount - count + i) %
		                   RecentSubmissionCount;
		const auto& debug = m_recent_submissions[index];
		if (!debug.valid || debug.tick > target_tick) {
			continue;
		}
		LOGD("  readback context: tick=%" PRIu64 " op=%u submit=%" PRIu64
		     " args=%u,%u,%u,%u,0x%016" PRIx64 " shader_hash=0x%016" PRIx64 "\n",
		     debug.tick, debug.op, debug.submit_id, debug.arg0, debug.arg1, debug.arg2,
		     debug.arg3, debug.arg4, debug.shader_hash);
	}
}

void CommandScheduler::PopPendingOperations() {
	{
		std::lock_guard lock(m_operation_mutex);
		if (m_pending_operations.empty()) {
			return;
		}
	}
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	#if defined(TRACY_ENABLE)
	if (m_tracy_gpu_context == nullptr && !m_tracy_gpu_profiling_checked &&
	    Config::ProfilerEnabled() &&
	    tracy::ProfilerAvailable()) {
		m_tracy_gpu_profiling_checked = true;
		const auto queue_families = m_graphics.physical_device.getQueueFamilyProperties();
		if (m_graphics.queue_family < queue_families.size() &&
		    queue_families[m_graphics.queue_family].timestampValidBits != 0) {
			Common::LockGuard lock(m_graphics.queue_mutex);
			m_tracy_gpu_context = tracy::CreateVkContext(
			    static_cast<VkInstance>(m_graphics.instance),
			    static_cast<VkPhysicalDevice>(m_graphics.physical_device),
			    static_cast<VkDevice>(m_graphics.device),
			    static_cast<VkQueue>(m_graphics.queue),
			    static_cast<VkCommandBuffer>(m_command.m_buffer),
			    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
			    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
			if (m_tracy_gpu_context != nullptr) {
				TracyVkContextName(static_cast<TracyVkCtx>(m_tracy_gpu_context),
				                   "Kyty host GPU", 13);
				LOGF("Tracy Vulkan GPU profiling enabled\n");
			}
		} else {
			LOGW("Tracy Vulkan GPU profiling unavailable: queue family has no timestamp support\n");
		}
	}
	#endif
	m_command.Begin();
	#if defined(TRACY_ENABLE)
	if (m_tracy_gpu_context != nullptr) {
		TracyVkCollect(static_cast<TracyVkCtx>(m_tracy_gpu_context),
		               static_cast<VkCommandBuffer>(m_command.Handle()));
		m_command.BeginGpuProfileZone(m_tracy_gpu_context);
	}
	#endif
	m_dynamic_state.Invalidate();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4, m_command.m_debug_shader_hash);
		if (result == vk::Result::eErrorDeviceLost) {
			ReportDeviceFault(graphics, tick);
		}
		if (m_command.m_mesh_fallback_debug_valid) {
			LOGE("Mesh fallback recorded in failed command buffer: submit=%" PRIu64
			     " groups=%ux%u indices=%u stride_dwords=%u output_bytes=%" PRIu64 "\n",
			     m_command.m_mesh_fallback_submit_id, m_command.m_mesh_fallback_groups_x,
			     m_command.m_mesh_fallback_groups_y, m_command.m_mesh_fallback_index_count,
			     m_command.m_mesh_fallback_stride_dwords, m_command.m_mesh_fallback_bytes);
		}
		if (m_last_mesh_fallback_submission.valid) {
			const auto& mesh = m_last_mesh_fallback_submission;
			LOGE("Most recent successfully submitted mesh fallback: tick=%" PRIu64
			     " submit=%" PRIu64 " groups=%ux%u indices=%u stride_dwords=%u "
			     "output_bytes=%" PRIu64 "\n",
			     mesh.tick, mesh.submit_id, mesh.groups_x, mesh.groups_y, mesh.index_count,
			     mesh.stride_dwords, mesh.output_bytes);
		}
		LOGE("Recent successful Vulkan queue submissions (oldest first):\n");
		for (size_t i = 0; i < m_recent_submission_count; i++) {
			const auto index = (m_recent_submission_next + RecentSubmissionCount -
			                    m_recent_submission_count + i) % RecentSubmissionCount;
			const auto& debug = m_recent_submissions[index];
			if (debug.mesh_fallback) {
				LOGE("  tick=%" PRIu64 " op=%u submit=%" PRIu64
				     " args=%u,%u,%u,%u,0x%016" PRIx64 " shader_hash=0x%016" PRIx64
				     " mesh_fallback=%" PRIu64
				     " groups=%ux%u indices=%u stride_dwords=%u output_bytes=%" PRIu64 "\n",
				     debug.tick, debug.op, debug.submit_id, debug.arg0, debug.arg1,
				     debug.arg2, debug.arg3, debug.arg4, debug.shader_hash, debug.mesh_submit_id,
				     debug.mesh_groups_x, debug.mesh_groups_y, debug.mesh_index_count,
				     debug.mesh_stride_dwords, debug.mesh_output_bytes);
			} else {
				LOGE("  tick=%" PRIu64 " op=%u submit=%" PRIu64
				     " args=%u,%u,%u,%u,0x%016" PRIx64 " shader_hash=0x%016" PRIx64 "\n",
				     debug.tick, debug.op, debug.submit_id, debug.arg0, debug.arg1,
				     debug.arg2, debug.arg3, debug.arg4, debug.shader_hash);
			}
		}
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	auto& debug = m_recent_submissions[m_recent_submission_next];
	debug = {};
	debug.valid       = true;
	debug.tick        = tick;
	debug.submit_id   = m_command.m_debug_submit_id;
	debug.shader_hash = m_command.m_debug_shader_hash;
	debug.op          = m_command.m_debug_op;
	debug.arg0        = m_command.m_debug_arg0;
	debug.arg1        = m_command.m_debug_arg1;
	debug.arg2        = m_command.m_debug_arg2;
	debug.arg3        = m_command.m_debug_arg3;
	debug.arg4        = m_command.m_debug_arg4;
	debug.mesh_fallback = m_command.m_mesh_fallback_debug_valid;
	if (m_command.m_mesh_fallback_debug_valid) {
		m_last_mesh_fallback_submission = {true,
		                                  tick,
		                                  m_command.m_mesh_fallback_submit_id,
		                                  m_command.m_mesh_fallback_groups_x,
		                                  m_command.m_mesh_fallback_groups_y,
		                                  m_command.m_mesh_fallback_index_count,
		                                  m_command.m_mesh_fallback_stride_dwords,
		                                  m_command.m_mesh_fallback_bytes};
		debug.mesh_submit_id       = m_command.m_mesh_fallback_submit_id;
		debug.mesh_groups_x        = m_command.m_mesh_fallback_groups_x;
		debug.mesh_groups_y        = m_command.m_mesh_fallback_groups_y;
		debug.mesh_index_count     = m_command.m_mesh_fallback_index_count;
		debug.mesh_stride_dwords   = m_command.m_mesh_fallback_stride_dwords;
		debug.mesh_output_bytes    = m_command.m_mesh_fallback_bytes;
	}
	m_recent_submission_next = (m_recent_submission_next + 1) % RecentSubmissionCount;
	m_recent_submission_count =
	    std::min(m_recent_submission_count + 1, RecentSubmissionCount);

	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
