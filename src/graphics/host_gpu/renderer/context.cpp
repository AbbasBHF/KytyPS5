#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>

#if defined(TRACY_ENABLE)
#  define TRACY_VK_USE_SYMBOL_TABLE
#  include <tracy/TracyVulkan.hpp>
#endif

namespace Libs::Graphics {

#if defined(TRACY_ENABLE)
struct TracyGpuCommandBufferZone {
	static constexpr tracy::SourceLocationData source_location {
	    "Emulator command buffer", "CommandBuffer", __FILE__,
	    static_cast<uint32_t>(__LINE__), 0};
	std::optional<tracy::VkCtxScope> scope;

	TracyGpuCommandBufferZone(TracyVkCtx context, VkCommandBuffer command_buffer) {
		scope.emplace(context, &source_location, command_buffer, true);
	}
};
#else
struct TracyGpuCommandBufferZone {};
#endif

CommandBuffer::~CommandBuffer() = default;

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	m_debug_op        = static_cast<uint32_t>(CommandBufferDebugOp::Unknown);
	m_debug_submit_id = 0;
	m_debug_arg0      = 0;
	m_debug_arg1      = 0;
	m_debug_arg2      = 0;
	m_debug_arg3      = 0;
	m_debug_arg4      = 0;
	m_debug_shader_hash = 0;
	m_mesh_fallback_debug_valid = false;
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::BeginGpuProfileZone(void* tracy_gpu_context) const {
#if defined(TRACY_ENABLE)
	if (tracy_gpu_context != nullptr && tracy::ProfilerAvailable()) {
		m_gpu_profile_zone = std::make_unique<TracyGpuCommandBufferZone>(
		    static_cast<TracyVkCtx>(tracy_gpu_context), static_cast<VkCommandBuffer>(Handle()));
	}
#else
	(void)tracy_gpu_context;
#endif
}

void CommandBuffer::End() const {
	EndRendering();
	m_gpu_profile_zone.reset();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
	m_debug_shader_hash = 0;
}

void CommandBuffer::SetMeshFallbackDebugInfo(uint64_t submit_id, uint32_t groups_x,
	                                         uint32_t groups_y, uint32_t index_count,
	                                         uint32_t group_stride_dwords, uint64_t bytes) {
	m_mesh_fallback_debug_valid   = true;
	m_mesh_fallback_submit_id     = submit_id;
	m_mesh_fallback_groups_x      = groups_x;
	m_mesh_fallback_groups_y      = groups_y;
	m_mesh_fallback_index_count   = index_count;
	m_mesh_fallback_stride_dwords = group_stride_dwords;
	m_mesh_fallback_bytes         = bytes;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
