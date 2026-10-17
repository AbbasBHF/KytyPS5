#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t MeshArray(EmitterState& state, spv::StorageClass storage, uint32_t type, uint32_t count) {
	const auto array = state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	return state.builder.DefineGlobalVariable(TypePointer(state, storage, array), storage);
}

uint32_t MeshElement(EmitterState& state, uint32_t variable, spv::StorageClass storage,
                     uint32_t type, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, type), pointer,
	                          variable, index);
	return pointer;
}

uint32_t MeshLoad(EmitterState& state, uint32_t variable, spv::StorageClass storage, uint32_t type,
                  uint32_t index) {
	const auto pointer = MeshElement(state, variable, storage, type, index);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, value, pointer);
	return value;
}

uint32_t MeshOutputType(EmitterState& state, IR::StageOutputKind kind) {
	switch (kind) {
		case IR::StageOutputKind::Layer: return TypeU32(state);
		case IR::StageOutputKind::ClipDistance:
		case IR::StageOutputKind::CullDistance: return TypeF32(state);
		case IR::StageOutputKind::PointSize: return TypeF32(state);
		case IR::StageOutputKind::ViewportIndex: return TypeU32(state);
		case IR::StageOutputKind::Position:
		case IR::StageOutputKind::Parameter: return TypeF32Vector(state, 4);
		default: EXIT("unsupported mesh output kind=%u\n", static_cast<uint32_t>(kind));
	}
}

} // namespace

void DefineMeshOutputs(EmitterState& state, uint32_t clip_distance_count,
                       uint32_t cull_distance_count) {
	const auto& mesh = state.input_info.vertex->mesh;
	for (auto& output: state.outputs) {
		const auto type = MeshOutputType(state, output.kind);
		const bool clip = output.kind == IR::StageOutputKind::ClipDistance;
		const bool cull = output.kind == IR::StageOutputKind::CullDistance;
		auto& variable = clip ? state.clip_distance_variable
		                 : cull ? state.cull_distance_variable : output.variable_id;
		if (variable == 0 && !state.mesh_shader_fallback) {
			const auto element_type = clip || cull
			                              ? state.builder.Type(spv::OpTypeArray, type,
			                                                   ConstantU32(state, clip ? clip_distance_count
			                                                                           : cull_distance_count))
			                              : type;
			variable = MeshArray(state, spv::StorageClassOutput, element_type,
			                     output.kind == IR::StageOutputKind::Layer ? mesh.max_primitives
			                                                             : mesh.max_vertices);
			state.interface_variables.push_back(variable);
			state.builder.AddName(variable, output.debug_name.c_str());
			if (output.kind == IR::StageOutputKind::Parameter) {
				state.builder.AddAnnotation(spv::OpDecorate, variable,
				                            spv::DecorationLocation, output.location);
			} else {
				const auto builtin = clip ? spv::BuiltInClipDistance
				                     : cull ? spv::BuiltInCullDistance
				                     : output.kind == IR::StageOutputKind::Layer ? spv::BuiltInLayer
				                     : output.kind == IR::StageOutputKind::PointSize
				                         ? spv::BuiltInPointSize
				                     : output.kind == IR::StageOutputKind::ViewportIndex
				                         ? spv::BuiltInViewportIndex
				                         : spv::BuiltInPosition;
				state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn, builtin);
			}
			if (output.kind == IR::StageOutputKind::Layer) {
				state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationPerPrimitiveEXT);
			}
		}
		output.variable_id = state.mesh_shader_fallback ? 0u : variable;
		// Only Layer is read by another invocation, through the primitive's provoking vertex.
		const bool shared = output.kind == IR::StageOutputKind::Layer;
		output.mesh_data_variable =
		    MeshArray(state, shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, type,
		              shared ? mesh.max_vertices : state.lane_count);
	}
	state.mesh_allocation = MeshArray(state, spv::StorageClassWorkgroup, TypeU32(state), 2);
	state.mesh_primitive_data =
	    MeshArray(state, spv::StorageClassPrivate, TypeU32(state), state.lane_count);
	const auto primitive_storage = state.mesh_shader_fallback ? spv::StorageClassPrivate
	                                                          : spv::StorageClassOutput;
	state.mesh_primitives = MeshArray(state, primitive_storage, TypeU32Vector(state, 3),
	                                  mesh.max_primitives);
	state.mesh_cull = MeshArray(state, primitive_storage, TypeBool(state), mesh.max_primitives);
	if (!state.mesh_shader_fallback) {
		state.interface_variables.push_back(state.mesh_primitives);
		state.interface_variables.push_back(state.mesh_cull);
		state.builder.AddAnnotation(spv::OpDecorate, state.mesh_primitives, spv::DecorationBuiltIn,
		                            spv::BuiltInPrimitiveTriangleIndicesEXT);
		state.builder.AddAnnotation(spv::OpDecorate, state.mesh_cull, spv::DecorationBuiltIn,
		                            spv::BuiltInCullPrimitiveEXT);
		state.builder.AddAnnotation(spv::OpDecorate, state.mesh_cull,
		                            spv::DecorationPerPrimitiveEXT);
	}
}

uint32_t MeshOutputPointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index) {
	const auto output = std::ranges::find_if(state.outputs, [=](const OutputBinding& binding) {
		return binding.kind == kind && binding.index == index;
	});
	if (output == state.outputs.end()) {
		EXIT("mesh export has no output binding: kind=%u index=%u\n", static_cast<uint32_t>(kind),
		     index);
	}
	const bool shared = kind == IR::StageOutputKind::Layer;
	return MeshElement(
	    state, output->mesh_data_variable,
	    shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, MeshOutputType(state, kind),
	    shared ? EmitLocalInvocationIndex(state) : ConstantU32(state, state.lane_half));
}

uint32_t MeshPrimitivePointer(EmitterState& state) {
	return MeshElement(state, state.mesh_primitive_data, spv::StorageClassPrivate, TypeU32(state),
	                   ConstantU32(state, state.lane_half));
}

void EmitMeshAllocate(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto first = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first,
	                          EmitLocalInvocationIndex(state), ConstantU32(state, 0));
	EmitIfCondition(state, first, [&] {
		const auto allocation = ctx.Arg(inst, 0);
		for (uint32_t field = 0; field < 2; field++) {
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), value, allocation,
			                          ConstantU32(state, field * 12u),
			                          ConstantU32(state, field == 0 ? 10u : 11u));
			const auto pointer =
			    MeshElement(state, state.mesh_allocation, spv::StorageClassWorkgroup,
			                TypeU32(state), ConstantU32(state, field));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		}
	});
}

void EmitMeshEntryPoint(EmitterState& state) {
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.builder.AllocateId());
	state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
	                          state.mesh_guest_func);
	// All guest waves finish before the uniform Vulkan allocation and output stores.
	EmitBarrier(state);
	const auto vertices   = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 0));
	const auto primitives = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 1));
	if (!state.mesh_shader_fallback) {
		state.builder.AddFunction(spv::OpSetMeshOutputsEXT, vertices,
		                          primitives); // OpSetMeshOutputsEXT
	} else {
		// The fallback buffer uses a fixed per-workgroup slot:
		// [vertex count, primitive count, vertex records, primitive records].
		const auto push_load = [&](uint32_t dword) {
			const auto pointer = state.builder.AllocateId();
			const auto value   = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state),
			                          pointer, state.push_constant_variable,
			                          ConstantU32(state, 0), ConstantU32(state, dword));
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
			return value;
		};
		const auto low       = push_load(6);
		const auto high      = push_load(7);
		const auto low64     = state.builder.AllocateId();
		const auto high64    = state.builder.AllocateId();
		const auto high_shift = state.builder.AllocateId();
		const auto base      = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpUConvert, TypeU64(state), low64, low);
		state.builder.AddFunction(spv::OpUConvert, TypeU64(state), high64, high);
		state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU64(state), high_shift, high64,
		                          ConstantU64(state, 32));
		state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), base, high_shift, low64);
		const auto group_x  = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
		const auto group_y  = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
		const auto groups_x = push_load(8);
		const auto row      = EmitBinaryU32(state, spv::OpIMul, group_y, groups_x);
		const auto group    = EmitAddU32(state, row, group_x);
		const auto group64  = state.builder.AllocateId();
		const auto slot64   = state.builder.AllocateId();
		const auto slot     = push_load(9);
		const auto words    = state.builder.AllocateId();
		const auto bytes    = state.builder.AllocateId();
		const auto group_offset = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpUConvert, TypeU64(state), group64, group);
		state.builder.AddFunction(spv::OpUConvert, TypeU64(state), slot64, slot);
		state.builder.AddFunction(spv::OpIMul, TypeU64(state), words, group64, slot64);
		state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU64(state), bytes, words,
		                          ConstantU64(state, 2));
		state.builder.AddFunction(spv::OpIAdd, TypeU64(state), group_offset, base, bytes);
		const auto store_word = [&](uint32_t word_offset, uint32_t value) {
			const auto offset64 = state.builder.AllocateId();
			const auto offset_bytes = state.builder.AllocateId();
			const auto address = state.builder.AllocateId();
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpUConvert, TypeU64(state), offset64, word_offset);
			state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU64(state), offset_bytes,
			                          offset64, ConstantU64(state, 2));
			state.builder.AddFunction(spv::OpIAdd, TypeU64(state), address, group_offset,
			                          offset_bytes);
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			                          address);
			state.builder.AddFunction(spv::OpStore, pointer, value,
			                          spv::MemoryAccessAlignedMask, uint32_t {4});
		};
		const auto first_lane = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first_lane,
		                          EmitLocalInvocationIndex(state), ConstantU32(state, 0));
		EmitIfCondition(state, first_lane, [&] {
			store_word(ConstantU32(state, 0), vertices);
			store_word(ConstantU32(state, 1), primitives);
		});
		const auto output_count = static_cast<uint32_t>(state.outputs.size());
		const auto layout = ShaderMeshFallbackLayout::Create(
		    state.input_info.vertex->mesh.max_vertices,
		    state.input_info.vertex->mesh.max_primitives, output_count);
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.lane_half = half;
			const auto index = EmitLocalInvocationIndex(state);
			const auto is_vertex = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_vertex, index, vertices);
			EmitIfCondition(state, is_vertex, [&] {
				for (uint32_t output_index = 0; output_index < output_count; output_index++) {
					const auto& output = state.outputs[output_index];
					if (output.kind == IR::StageOutputKind::Layer) continue;
					const auto value = MeshLoad(state, output.mesh_data_variable,
					                           spv::StorageClassPrivate, MeshOutputType(state, output.kind),
					                           ConstantU32(state, half));
					const auto base_word = EmitAddU32(
					    state, ConstantU32(state, 2u + output_index * 4u),
				    EmitBinaryU32(state, spv::OpIMul, index,
				                  ConstantU32(state, layout.vertex_stride_dwords)));
					const bool scalar = output.kind == IR::StageOutputKind::ClipDistance ||
					                    output.kind == IR::StageOutputKind::CullDistance;
					const auto components = scalar ? 1u : 4u;
					for (uint32_t component = 0; component < components; component++) {
						auto word = value;
						if (!scalar) {
							word = state.builder.AllocateId();
							state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), word,
							                          value, component);
						}
						const auto bits = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, word);
						const auto word_offset = component == 0 ? base_word
						    : EmitAddU32(state, base_word, ConstantU32(state, component));
						store_word(word_offset, bits);
					}
				}
			});
			const auto is_primitive = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_primitive, index,
			                          primitives);
			EmitIfCondition(state, is_primitive, [&] {
				const auto packed = MeshLoad(state, state.mesh_primitive_data,
				                             spv::StorageClassPrivate, TypeU32(state),
				                             ConstantU32(state, half));
				const auto primitive_offset = EmitAddU32(
				    state, ConstantU32(state, layout.primitive_data_offset),
				    EmitBinaryU32(state, spv::OpIMul, index,
				                  ConstantU32(state, layout.primitive_stride_dwords)));
				store_word(primitive_offset, packed);
				uint32_t vertices_index[3] {};
				for (uint32_t component = 0; component < 3; component++) {
					vertices_index[component] = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state),
					                          vertices_index[component], packed,
					                          ConstantU32(state, component * 10u),
					                          ConstantU32(state, 10));
				}
				for (uint32_t output_index = 0; output_index < output_count; output_index++) {
					const auto& output = state.outputs[output_index];
					if (output.kind != IR::StageOutputKind::Layer) continue;
					const auto provoking_vertex = vertices_index[
					    state.input_info.vertex->mesh.provoking_vertex];
					const auto provoking_vertex_valid = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpULessThan, TypeBool(state), provoking_vertex_valid, provoking_vertex,
					    ConstantU32(state, state.input_info.vertex->mesh.max_vertices));
					const auto safe_provoking_vertex = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpSelect, TypeU32(state), safe_provoking_vertex,
					                          provoking_vertex_valid, provoking_vertex,
					                          ConstantU32(state, 0));
					const auto layer = MeshLoad(
					    state, output.mesh_data_variable, spv::StorageClassWorkgroup, TypeU32(state),
					    safe_provoking_vertex);
					store_word(EmitAddU32(state, primitive_offset,
					                      ConstantU32(state, output_index + 1u)), layer);
				}
			});
		}
	}
	if (!state.mesh_shader_fallback) {
	for (uint32_t half = 0; half < state.lane_count; half++) {
		state.lane_half      = half;
		const auto index     = EmitLocalInvocationIndex(state);
		const auto is_vertex = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_vertex, index, vertices);
		EmitIfCondition(state, is_vertex, [&] {
			for (const auto& output: state.outputs) {
				if (output.kind == IR::StageOutputKind::Layer) {
					continue;
				}
				const auto type  = MeshOutputType(state, output.kind);
				const auto value =
				    MeshLoad(state, output.mesh_data_variable, spv::StorageClassPrivate, type,
				             ConstantU32(state, half));
				uint32_t pointer;
				if (output.kind == IR::StageOutputKind::ClipDistance ||
				    output.kind == IR::StageOutputKind::CullDistance) {
					pointer = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, type), pointer,
					    output.variable_id, index, ConstantU32(state, output.index));
				} else {
					pointer = MeshElement(state, output.variable_id, spv::StorageClassOutput, type, index);
				}
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
		});
		const auto is_primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_primitive, index,
		                          primitives);
		EmitIfCondition(state, is_primitive, [&] {
			const auto packed = MeshLoad(state, state.mesh_primitive_data, spv::StorageClassPrivate,
			                             TypeU32(state), ConstantU32(state, half));
			uint32_t   vertex[3] {};
			for (uint32_t component = 0; component < 3; component++) {
				vertex[component] = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpBitFieldUExtract, TypeU32(state), vertex[component], packed,
				    ConstantU32(state, component * 10u), ConstantU32(state, 10));
			}
			const auto triangle = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 3), triangle,
			                          vertex[0], vertex[1], vertex[2]);
					const auto triangle_pointer =
			    MeshElement(state, state.mesh_primitives, spv::StorageClassOutput,
			                TypeU32Vector(state, 3), index);
			state.builder.AddFunction(spv::OpStore, triangle_pointer, triangle);
			const auto null_bit =
			    EmitBinaryU32(state, spv::OpBitwiseAnd, packed, ConstantU32(state, 0x80000000u));
			const auto culled = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), culled, null_bit,
			                          ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore,
			                          MeshElement(state, state.mesh_cull, spv::StorageClassOutput,
			                                      TypeBool(state), index),
			                          culled);
			for (const auto& output: state.outputs) {
				if (output.kind != IR::StageOutputKind::Layer) {
					continue;
				}
				const auto layer = MeshLoad(state, output.mesh_data_variable,
				                            spv::StorageClassWorkgroup, TypeU32(state),
				                            vertex[state.input_info.vertex->mesh.provoking_vertex]);
				const auto pointer = MeshElement(state, output.variable_id, spv::StorageClassOutput,
				                                 TypeU32(state), index);
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
		});
	}
	}
	state.lane_half = 0;
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

namespace {

uint32_t MeshFallbackPushLoad(EmitterState& state, uint32_t dword) {
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, ConstantU32(state, 0),
	                          ConstantU32(state, dword));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t MeshFallbackAddress(EmitterState& state, uint32_t base, uint32_t dword_offset) {
	const auto offset64 = state.builder.AllocateId();
	const auto byte_offset = state.builder.AllocateId();
	const auto address = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUConvert, TypeU64(state), offset64, dword_offset);
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU64(state), byte_offset, offset64,
	                          ConstantU64(state, 2));
	state.builder.AddFunction(spv::OpIAdd, TypeU64(state), address, base, byte_offset);
	return address;
}

uint32_t MeshFallbackLoadWord(EmitterState& state, uint32_t address) {
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
	                          address);
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessAlignedMask, uint32_t {4});
	return value;
}

} // namespace

std::vector<uint32_t> EmitMeshFallbackVertexProgram(const IR::CompiledShaderInfo& mesh_program,
                                                     const ShaderMeshInputInfo& mesh) {
	IR::Program program;
	program.stage             = ShaderType::Vertex;
	program.wave_size         = mesh.wave_size;
	program.info.outputs      = mesh_program.info.outputs;
	program.info.inputs.push_back({IR::StageInputKind::VertexIndex, 0, 1, "gl_VertexIndex"});
	program.info.inputs.push_back({IR::StageInputKind::InstanceIndex, 0, 1, "gl_InstanceIndex"});
	ShaderVertexInputInfo vertex_input {};
	vertex_input.mesh = mesh;
	ShaderStageInputInfo stage_input {};
	stage_input.vertex = &vertex_input;
	EmitterState state(program, stage_input, false, true);
	DefineModule(state);
	state.builder.RequireCapability(spv::CapabilityInt64);
	state.builder.RequireCapability(spv::CapabilityPhysicalStorageBufferAddresses);
	state.builder.RequireExtension("SPV_KHR_physical_storage_buffer");

	const auto push_array = state.builder.DecoratedType(
	    spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, sizeof(uint32_t)}}},
	    TypeU32(state), ConstantU32(state, IR::PushData::DwordCount));
	const auto push_block = state.builder.DecoratedType(
	    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                        {spv::OpDecorate, {spv::DecorationBlock}}}, push_array);
	state.push_constant_variable = state.builder.DefineGlobalVariable(
	    TypePointer(state, spv::StorageClassPushConstant, push_block), spv::StorageClassPushConstant);
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.entry_label);
	const auto packed_primitive_variable = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpVariable,
	                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
	                          packed_primitive_variable, spv::StorageClassFunction);
	state.builder.AddFunction(spv::OpStore, packed_primitive_variable, ConstantU32(state, 0));

	const auto vertex_index = EmitInputComponentU32(state, IR::StageInputKind::VertexIndex, 0);
	const auto output_base_low = MeshFallbackPushLoad(state, 0);
	const auto output_base_high = MeshFallbackPushLoad(state, 1);
	const auto low64 = state.builder.AllocateId();
	const auto high64 = state.builder.AllocateId();
	const auto high_shift = state.builder.AllocateId();
	const auto output_base = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUConvert, TypeU64(state), low64, output_base_low);
	state.builder.AddFunction(spv::OpUConvert, TypeU64(state), high64, output_base_high);
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU64(state), high_shift, high64,
	                          ConstantU64(state, 32));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), output_base, low64, high_shift);
	const auto group_stride = MeshFallbackPushLoad(state, 2);
	const auto groups_x = MeshFallbackPushLoad(state, 3);
	const auto max_primitives = MeshFallbackPushLoad(state, 4);
	const auto primitive_vertices = EmitBinaryU32(state, spv::OpIMul, max_primitives,
	                                             ConstantU32(state, 3));
	const auto local_group = EmitBinaryU32(state, spv::OpUDiv, vertex_index, primitive_vertices);
	const auto instance_index = EmitInputComponentU32(state, IR::StageInputKind::InstanceIndex, 0);
	const auto instance_base = EmitBinaryU32(state, spv::OpIMul, instance_index, groups_x);
	const auto group = EmitAddU32(state, instance_base, local_group);
	const auto primitive_flat = EmitBinaryU32(state, spv::OpUDiv, vertex_index,
	                                         ConstantU32(state, 3));
	const auto local_primitive = EmitBinaryU32(state, spv::OpUMod, primitive_flat,
	                                          max_primitives);
	const auto corner = EmitBinaryU32(state, spv::OpUMod, vertex_index, ConstantU32(state, 3));
	const auto group_words = EmitBinaryU32(state, spv::OpIMul, group, group_stride);
	const auto group_address = MeshFallbackAddress(state, output_base, group_words);
	const auto group_address_plus_one = MeshFallbackAddress(state, group_address, ConstantU32(state, 1));
	const auto vertex_count = MeshFallbackLoadWord(state, group_address);
	const auto primitive_count = MeshFallbackLoadWord(state, group_address_plus_one);
	const auto output_count = static_cast<uint32_t>(mesh_program.info.outputs.size());
	const auto layout = ShaderMeshFallbackLayout::Create(mesh.max_vertices, mesh.max_primitives,
	                                                     output_count);
	const auto primitive_start = EmitAddU32(
	    state, ConstantU32(state, layout.primitive_data_offset),
	    EmitBinaryU32(state, spv::OpIMul, local_primitive,
	                  ConstantU32(state, layout.primitive_stride_dwords)));
	const auto in_primitive_range = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), in_primitive_range,
	                          local_primitive, primitive_count);
	EmitIfCondition(state, in_primitive_range, [&] {
		const auto packed_address = MeshFallbackAddress(state, group_address, primitive_start);
		const auto packed = MeshFallbackLoadWord(state, packed_address);
		state.builder.AddFunction(spv::OpStore, packed_primitive_variable, packed);
	});
	const auto packed = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), packed, packed_primitive_variable);
	const auto corner_shift = EmitBinaryU32(state, spv::OpIMul, corner, ConstantU32(state, 10));
	const auto triangle_vertex = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), triangle_vertex, packed,
	                          corner_shift, ConstantU32(state, 10));
	const auto in_range = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), in_range, triangle_vertex,
	                          vertex_count);
	const auto culled_bits = EmitBinaryU32(state, spv::OpBitwiseAnd, packed,
	                                      ConstantU32(state, 0x80000000u));
	const auto culled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), culled, culled_bits,
	                          ConstantU32(state, 0));
	const auto not_culled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), not_culled, culled);
	const auto live_primitive = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), live_primitive,
	                          in_primitive_range, not_culled);
	const auto valid = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), valid, live_primitive, in_range);
	const auto safe_vertex = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), safe_vertex, in_range,
	                          triangle_vertex, ConstantU32(state, 0));
	const auto vertex_start = EmitAddU32(
	    state, ConstantU32(state, 2u),
	    EmitBinaryU32(state, spv::OpIMul, safe_vertex,
	                  ConstantU32(state, layout.vertex_stride_dwords)));
	const auto has_vertex = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), has_vertex,
	                          ConstantU32(state, 0), vertex_count);
	const auto output_zero = ConstantU32(state, 0);
	for (uint32_t output_index = 0; output_index < output_count; output_index++) {
		const auto& output = mesh_program.info.outputs[output_index];
		const auto binding = std::ranges::find_if(state.outputs, [&](const OutputBinding& item) {
			return item.kind == output.kind && item.index == output.index &&
			       item.location == output.location;
		});
		EXIT_IF(binding == state.outputs.end());
		if (output.kind == IR::StageOutputKind::Position ||
		    output.kind == IR::StageOutputKind::Parameter) {
			uint32_t components[4] {};
			for (uint32_t component = 0; component < 4; component++) {
				const auto word_offset = EmitAddU32(
				    state, vertex_start, ConstantU32(state, output_index * 4u + component));
				const auto safe_word_offset = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), safe_word_offset, has_vertex,
				                          word_offset, ConstantU32(state, 0));
				const auto address = MeshFallbackAddress(state, group_address, safe_word_offset);
				const auto bits = MeshFallbackLoadWord(state, address);
				const auto fallback_bits = output.kind == IR::StageOutputKind::Position && component < 3
				                               ? ConstantU32(state, 0x40000000u)
				                               : output.kind == IR::StageOutputKind::Position && component == 3
				                                     ? ConstantU32(state, 0x3f800000u)
				                                     : output_zero;
				components[component] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), components[component], valid,
				                          bits, fallback_bits);
			}
			const auto f32 = TypeF32(state);
			uint32_t floats[4] {};
			for (uint32_t component = 0; component < 4; component++) {
				floats[component] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, f32, floats[component], components[component]);
			}
			const auto vector = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector,
			                          floats[0], floats[1], floats[2], floats[3]);
			uint32_t pointer = binding->variable_id;
			if (output.kind == IR::StageOutputKind::Position) {
				pointer = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput,
				                                   TypeF32Vector(state, 4)),
				    pointer, binding->variable_id, ConstantU32(state, 0));
			}
			state.builder.AddFunction(spv::OpStore, pointer, vector);
		} else if (output.kind == IR::StageOutputKind::Layer) {
			const auto candidate_offset = EmitAddU32(state, primitive_start,
			                                          ConstantU32(state, output_index + 1u));
			const auto layer_offset = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), layer_offset,
			                          in_primitive_range, candidate_offset, ConstantU32(state, 0));
			const auto layer_address = MeshFallbackAddress(state, group_address, layer_offset);
			const auto layer = MeshFallbackLoadWord(state, layer_address);
			const auto selected = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, valid, layer,
			                          output_zero);
			state.builder.AddFunction(spv::OpStore, binding->variable_id, selected);
		} else if (output.kind == IR::StageOutputKind::ClipDistance ||
		           output.kind == IR::StageOutputKind::CullDistance ||
		           output.kind == IR::StageOutputKind::PointSize ||
		           output.kind == IR::StageOutputKind::ViewportIndex) {
			const auto word_offset = EmitAddU32(
			    state, vertex_start, ConstantU32(state, output_index * 4u));
			const auto safe_word_offset = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), safe_word_offset, has_vertex,
			                          word_offset, ConstantU32(state, 0));
			const auto address = MeshFallbackAddress(state, group_address, safe_word_offset);
			const auto bits = MeshFallbackLoadWord(state, address);
			const auto is_distance = output.kind == IR::StageOutputKind::ClipDistance ||
			                         output.kind == IR::StageOutputKind::CullDistance;
			if (is_distance || output.kind == IR::StageOutputKind::PointSize) {
				const auto value = state.builder.AllocateId();
				const auto selected = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeF32(state), value, bits);
				state.builder.AddFunction(spv::OpSelect, TypeF32(state), selected, valid, value,
				                          ConstantF32Value(state, 1.0f));
				if (is_distance) {
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
				    pointer, binding->variable_id, ConstantU32(state, output.index));
					state.builder.AddFunction(spv::OpStore, pointer, selected);
				} else {
					state.builder.AddFunction(spv::OpStore, binding->variable_id, selected);
				}
			} else {
				const auto selected = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, valid, bits,
				                          output_zero);
				state.builder.AddFunction(spv::OpStore, binding->variable_id, selected);
			}
		} else {
			EXIT("mesh fallback vertex stage cannot forward output kind=%u\n",
			     static_cast<uint32_t>(output.kind));
		}
	}
	if (state.invalid_position_clip_distance != UINT32_MAX) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
		    pointer, state.clip_distance_variable,
		    ConstantU32(state, state.invalid_position_clip_distance));
		state.builder.AddFunction(spv::OpStore, pointer, ConstantF32Value(state, 1.0f));
	}
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
	state.builder.AddEntryPoint(spv::ExecutionModelVertex, state.main_func, "main",
	                            state.interface_variables);
	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
