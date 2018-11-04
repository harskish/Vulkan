/*
* Vulkan Example - NVIDIA hardware accelerated ray tracing
*
* Copyright (C) 2018 by Erik Härkönen
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>
#include <numeric>

constexpr int RT_STAGE_COUNT = 3; // [rgen, miss, chit]

struct VertexModel {
    glm::vec4 pos;
    glm::vec4 normal;
    glm::vec4 color;
};

vks::model::VertexLayout vertexLayoutModel{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,
    vks::model::VERTEX_COMPONENT_COLOR,
    vks::model::VERTEX_COMPONENT_DUMMY_FLOAT,
} };

class VulkanExample : public vkx::ExampleBase {
private:
    
    // https://github.com/KhronosGroup/Vulkan-Docs/issues/813
    struct InstanceNVX {
        float transform[12];
        uint32_t instanceID : 24;
        uint32_t instanceMask : 8;
        uint32_t instanceContributionToHitGroupIndex : 24;
        uint32_t flags : 8;
        uint64_t accelerationStructureHandle;
    };
    
    vk::DispatchLoaderDynamic loaderNVX;
    vks::Image textureRaytracingTarget;
    uint32_t rtDevMaxRecursionDepth = 0;
    uint32_t rtDevShaderHeaderSize = 0;
    uint32_t numMeshesNVX = 1;
    uint32_t numInstancesNVX = 1;
    std::vector<InstanceNVX> instances;
    std::vector<vk::GeometryNVX> geometries;
    vks::Buffer transform3x4;
    vks::Buffer scratchMem;
    vks::Buffer instancesNVX;
    vks::Buffer shaderBindingTable;
    vks::Buffer topLevelAccBuff;
    vks::Buffer bottomLevelAccBuff;
    vk::AccelerationStructureNVX topHandle;
    vk::AccelerationStructureNVX bottomHandle;
    const int rtStageCount = 3; // [rgen, miss, chit]

public:
    struct {
        vks::model::Model quad; // for fragment shader
        vks::model::Model rtMesh; // for actual raytracing
    } meshes;

    vks::Buffer uniformDataRaytracing;

    // Order by size to avoid alignment mismatches between host and device
    struct UboCompute {
        glm::mat4 invR;
        glm::vec4 camPos = glm::vec4(0.5f, 0.0f, 0.0f, 0.0f);
        glm::vec4 lightPos;
        float aspectRatio;
        float fov = 90.0f;
    } uboRT;

    struct {
        vk::Pipeline display;
        vk::Pipeline raytracing;
    } pipelines;

    int vertexBufferSize;

    vk::Queue raytracingQueue;
    vk::CommandBuffer raytracingCmdBuffer;
    vk::PipelineLayout raytracingPipelineLayout;
    vk::DescriptorSet raytracingDescriptorSet;
    vk::DescriptorSetLayout raytracingDescriptorSetLayout;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSetPostCompute;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 1.0f;
        camera.setPosition(glm::vec3(uboRT.camPos));
        camera.setRotation(glm::vec3(0.0f, 90.0f, 0.0f));
        camera.setPerspective(uboRT.fov, (float)size.width / (float)size.height, 0.1f, 64.0f);

        title = "Vulkan Example - NVX raytracing";
        uboRT.aspectRatio = (float)size.width / (float)size.height;
        paused = false;
        timerSpeed *= 0.5f;
    }

    ~VulkanExample() {
        device.destroyPipeline(pipelines.display);
        device.destroyPipeline(pipelines.raytracing);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroyPipelineLayout(raytracingPipelineLayout);
        device.destroyDescriptorSetLayout(raytracingDescriptorSetLayout);

        meshes.quad.destroy();
        meshes.rtMesh.destroy();
        uniformDataRaytracing.destroy();

        device.freeCommandBuffers(cmdPool, raytracingCmdBuffer);

        textureRaytracingTarget.destroy();
    }

    void getRTDeviceInfo() {
        auto props = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRaytracingPropertiesNVX>();
        auto devprops = props.get<vk::PhysicalDeviceProperties2>();
        auto rtprops = props.get<vk::PhysicalDeviceRaytracingPropertiesNVX>();

        rtDevMaxRecursionDepth = rtprops.maxRecursionDepth;
        rtDevShaderHeaderSize = rtprops.shaderHeaderSize;

        std::cout << "Raytracing device (" << devprops.properties.deviceName << "):" << std::endl
            << "shaderHeaderSize: " << "\t" << rtprops.shaderHeaderSize << std::endl
            << "maxRecursionDepth: " << "\t" << rtprops.maxRecursionDepth << std::endl
            << "maxGeometryCount: " << "\t" << rtprops.maxGeometryCount << std::endl
            << std::endl;
    }

    // Prepare a texture target that is used to store raytracing calculations
    void prepareTextureTarget(vks::Image& tex, uint32_t width, uint32_t height, vk::Format format) {
        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& setupCmdBuffer) {
            // Get device properties for the requested texture format
            vk::FormatProperties formatProperties;
            formatProperties = physicalDevice.getFormatProperties(format);
            // Check if requested image format supports image storage operations
            assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);

            // Prepare blit target texture
            tex.extent.width = width;
            tex.extent.height = height;

            vk::ImageCreateInfo imageCreateInfo;
            imageCreateInfo.imageType = vk::ImageType::e2D;
            imageCreateInfo.format = format;
            imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
            imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
            imageCreateInfo.initialLayout = vk::ImageLayout::ePreinitialized;
            // vk::Image will be sampled in the fragment shader and used as storage target in the raytracing stage
            imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
            tex = context.createImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
            context.setImageLayout(setupCmdBuffer, tex.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::ePreinitialized, vk::ImageLayout::eGeneral);

            // Create sampler
            vk::SamplerCreateInfo sampler;
            sampler.magFilter = vk::Filter::eLinear;
            sampler.minFilter = vk::Filter::eLinear;
            sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
            sampler.addressModeU = vk::SamplerAddressMode::eRepeat;
            sampler.addressModeV = sampler.addressModeU;
            sampler.addressModeW = sampler.addressModeU;
            sampler.mipLodBias = 0.0f;
            sampler.maxAnisotropy = 0;
            sampler.compareOp = vk::CompareOp::eNever;
            sampler.minLod = 0.0f;
            sampler.maxLod = 0.0f;
            sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
            tex.sampler = device.createSampler(sampler);

            // Create image view
            vk::ImageViewCreateInfo view;
            view.viewType = vk::ImageViewType::e2D;
            view.format = format;
            view.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
            view.image = tex.image;
            tex.view = device.createImageView(view);
        });
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        // vk::Image memory barrier to make sure that raytracing
        // shader writes are finished before sampling
        // from the texture
        vk::ImageMemoryBarrier imageMemoryBarrier;
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eGeneral;
        imageMemoryBarrier.newLayout = vk::ImageLayout::eGeneral;
        imageMemoryBarrier.image = textureRaytracingTarget.image;
        imageMemoryBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eInputAttachmentRead;
        cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, vk::DependencyFlags(), nullptr, nullptr,
            imageMemoryBarrier);
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
        // Display ray traced image generated as a full screen quad
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSetPostCompute, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.display);
        cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
    }

    void updateRaytracingCommandBuffer() {
        vk::CommandBufferBeginInfo cmdBufInfo;
        raytracingCmdBuffer.begin(cmdBufInfo);
        raytracingCmdBuffer.bindPipeline(vk::PipelineBindPoint::eRaytracingNVX, pipelines.raytracing);
        raytracingCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRaytracingNVX, raytracingPipelineLayout, 0, raytracingDescriptorSet, nullptr);
        
        auto stride = rtDevShaderHeaderSize;
        raytracingCmdBuffer.traceRaysNVX(
            shaderBindingTable.buffer,
            0, // raygen
            shaderBindingTable.buffer, 1 * rtDevShaderHeaderSize, stride, // miss
            shaderBindingTable.buffer, 2 * rtDevShaderHeaderSize, stride, // chit
            textureRaytracingTarget.extent.width,
            textureRaytracingTarget.extent.height,
            loaderNVX);

        raytracingCmdBuffer.end();
    }

    void traceRays() {
        vk::SubmitInfo raytracingSubmitInfo;
        raytracingSubmitInfo.commandBufferCount = 1;
        raytracingSubmitInfo.pCommandBuffers = &raytracingCmdBuffer;
        raytracingQueue.submit(raytracingSubmitInfo, nullptr);
        raytracingQueue.waitIdle();
    }

    // Setup vertices for a single uv-mapped quad
    void loadGeometry() {
        // Setup geometry for raytraing
        vks::model::ModelCreateInfo modelCreateInfo;
        modelCreateInfo.scale = glm::vec3(0.1f, -0.1f, 0.1f);
        modelCreateInfo.uvscale = glm::vec2(1.0f);
        modelCreateInfo.center = glm::vec3(0.0f, 0.0f, 0.0f);
        meshes.rtMesh.loadFromFile(context, getAssetPath() + "models/sibenik/sibenik.dae", vertexLayoutModel, modelCreateInfo);
        const vk::IndexType meshIndexType = vk::IndexType::eUint32; // loadFromFile uses U32

        // Identity transformation for all meshes
        std::vector<float> Id3x4 = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f
        };
        transform3x4 = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eUniformBuffer, Id3x4);

        const int numMeshes = 1;
        for (int i = 0; i < numMeshes; i++) {
            auto numVert = meshes.rtMesh.vertexCount;
            auto numInd = meshes.rtMesh.indexCount;
            auto strideVert = sizeof(VertexModel);
            auto tris = vk::GeometryTrianglesNVX(meshes.rtMesh.vertices.buffer, 0, numVert, strideVert, vk::Format::eR32G32B32Sfloat, meshes.rtMesh.indices.buffer, 0, numInd, meshIndexType,
                transform3x4.buffer, 0);
            auto geomData = vk::GeometryDataNVX(tris);                                          // union of tri and aabb, data read based on geometryTypeNVX
            auto geomFlags = vk::GeometryFlagBitsNVX::eOpaque;                                  // hits cannot be rejected (anyHit shader never run)
            auto geom = vk::GeometryNVX(vk::GeometryTypeNVX::eTriangles, geomData, geomFlags);  // type is triangles
            geometries.push_back(geom);
        }

        // Setup quad for drawing resulting image form raytracing pass
        struct VertexQuad {
            float pos[3];
            float uv[3];
        };
        const float dim = 1.0f;
        std::vector<VertexQuad> vertexBuffer = { { { dim,  dim, 0.0f }, { 1.0f, 1.0f } },
                                                { { -dim,  dim, 0.0f }, { 0.0f, 1.0f } },
                                                { { -dim, -dim, 0.0f }, { 0.0f, 0.0f } },
                                                 { { dim, -dim, 0.0f }, { 1.0f, 0.0f } } };

        meshes.quad.vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    // Rasterization
    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Fragment shader image sampler
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    // Rasterization
    void setupDescriptorSet() {
        descriptorSetPostCompute = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        updateDescriptorSet();
    }

    void updateDescriptorSet() {
        // vk::Image descriptor for the color map texture
        vk::DescriptorImageInfo texDescriptor{ textureRaytracingTarget.sampler, textureRaytracingTarget.view, vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Fragment shader texture sampler
            { descriptorSetPostCompute, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    // Create a separate command buffer for raytracing commands
    void createRaytracingCommandBuffer() { raytracingCmdBuffer = device.allocateCommandBuffers({ cmdPool, vk::CommandBufferLevel::ePrimary, 1 })[0]; }

    void preparePipelines() {
        vks::model::VertexLayout vertexLayoutQuad{ {
            vks::model::VERTEX_COMPONENT_POSITION,
            vks::model::VERTEX_COMPONENT_COLOR,
        } };
        // Display pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout, renderPass };
        pipelineCreator.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineCreator.vertexInputState.appendVertexLayout(vertexLayoutQuad);
        pipelineCreator.loadShader(getAssetPath() + "shaders/raytracing/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(getAssetPath() + "shaders/raytracing/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.display = pipelineCreator.create(context.pipelineCache);
    }

    void buildAccelerationStructure() {
        auto buildFlags = vk::BuildAccelerationStructureFlagBitsNVX::ePreferFastTrace;
        auto compactedSize = vk::DeviceSize(0);

        // Element counts
        uint32_t instanceCount = numInstancesNVX; // top-level, number of instances of bottom-level structures
        uint32_t geometryCount = numMeshesNVX;    // bottom-level

        auto topInfo = vk::AccelerationStructureCreateInfoNVX(vk::AccelerationStructureTypeNVX::eTopLevel, buildFlags, 0, instanceCount, 0);
        auto bottomInfo = vk::AccelerationStructureCreateInfoNVX(vk::AccelerationStructureTypeNVX::eBottomLevel, buildFlags, compactedSize, 0, geometryCount,
            geometries.data());

        topHandle = device.createAccelerationStructureNVX(topInfo, nullptr, loaderNVX);
        bottomHandle = device.createAccelerationStructureNVX(bottomInfo, nullptr, loaderNVX);

        // Get required sizes
        auto topLevelReq = vk::AccelerationStructureMemoryRequirementsInfoNVX(topHandle);
        auto scratchReqTop = vk::MemoryRequirements2();
        auto storageReqTop = vk::MemoryRequirements2();
        device.getAccelerationStructureScratchMemoryRequirementsNVX(&topLevelReq, &scratchReqTop, loaderNVX);
        device.getAccelerationStructureMemoryRequirementsNVX(&topLevelReq, &storageReqTop, loaderNVX);

        auto bottomLevelReq = vk::AccelerationStructureMemoryRequirementsInfoNVX(bottomHandle);
        auto scratchReqBot = vk::MemoryRequirements2();
        auto storageReqBot = vk::MemoryRequirements2();
        device.getAccelerationStructureScratchMemoryRequirementsNVX(&bottomLevelReq, &scratchReqBot, loaderNVX);
        device.getAccelerationStructureMemoryRequirementsNVX(&bottomLevelReq, &storageReqBot, loaderNVX);

        std::cout << "Top level storage requirement: " << storageReqTop.memoryRequirements.size << "B" << std::endl;
        std::cout << "Bottom level storage requirement: " << storageReqBot.memoryRequirements.size << "B" << std::endl;

        // Scratch mem upper bound size
        vk::MemoryRequirements scratchReqs = scratchReqBot.memoryRequirements;
        scratchReqs.size = std::max(scratchReqTop.memoryRequirements.size, scratchReqBot.memoryRequirements.size);

        // TODO: set alignments and type bits as well!
        //scratchMem = context.createBuffer(vk::BufferUsageFlagBits::eRaytracingNVX, vk::MemoryPropertyFlagBits::eDeviceLocal, scratchReqs.size);
        scratchMem = context.createBufferAligned(vk::BufferUsageFlagBits::eRaytracingNVX, vk::MemoryPropertyFlagBits::eDeviceLocal, scratchReqs);
        
        topLevelAccBuff = context.createBufferAligned(vk::BufferUsageFlagBits::eRaytracingNVX, vk::MemoryPropertyFlagBits::eDeviceLocal, storageReqTop.memoryRequirements);
        bottomLevelAccBuff = context.createBufferAligned(vk::BufferUsageFlagBits::eRaytracingNVX, vk::MemoryPropertyFlagBits::eDeviceLocal, storageReqBot.memoryRequirements);

        // Attach memory to acceleratio structures
        auto topMemoryInfo = vk::BindAccelerationStructureMemoryInfoNVX(topHandle, topLevelAccBuff.memory, 0, 0, nullptr);
        device.bindAccelerationStructureMemoryNVX(topMemoryInfo, loaderNVX);
        auto bottomMemoryInfo = vk::BindAccelerationStructureMemoryInfoNVX(bottomHandle, bottomLevelAccBuff.memory, 0, 0, nullptr);
        device.bindAccelerationStructureMemoryNVX(bottomMemoryInfo, loaderNVX);

        // Create instances
        for (int i = 0; i < numInstancesNVX; i++) {
            InstanceNVX instance {}; // zero init
            float identity[12] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f
            };
            memcpy(&instance.transform, identity, sizeof(identity));
            instance.instanceID = 0;
            instance.instanceMask = 0xff;
            instance.instanceContributionToHitGroupIndex = 0;
            instance.flags = (uint32_t)vk::GeometryInstanceFlagBitsNVX::eTriangleCullDisable;
            if (vk::Result::eSuccess != device.getAccelerationStructureHandleNVX(bottomHandle, sizeof(uint64_t), &instance.accelerationStructureHandle, loaderNVX)) {
                throw std::exception();
            }
            instances.push_back(instance);
        }

        // Create instance buffer
        instancesNVX = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eRaytracingNVX, instances);

        // Build
        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& commandBuffer) {
            auto readWriteBits = vk::AccessFlagBits::eAccelerationStructureReadNVX | vk::AccessFlagBits::eAccelerationStructureWriteNVX;
            auto bottomBuiltBarrier = vk::MemoryBarrier(readWriteBits, readWriteBits); // src, dst

            // Build bottom level BVH
            commandBuffer.buildAccelerationStructureNVX(
                vk::AccelerationStructureTypeNVX::eBottomLevel,
                0,                   // instance count
                nullptr,             // instancedata
                0,                   // instanceoffset
                numMeshesNVX,        // geomcount
                geometries.data(),   // geometries
                vk::BuildAccelerationStructureFlagBitsNVX::ePreferFastTrace, vk::Bool32(false), bottomHandle, nullptr,
                scratchMem.buffer,
                0,  // scratch offset
                loaderNVX
            );

            // Wait for bottom level build to finish
            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eRaytracingNVX, vk::PipelineStageFlagBits::eRaytracingNVX,
                vk::DependencyFlagBits::eDeviceGroup, bottomBuiltBarrier, nullptr, nullptr, loaderNVX);

            // Build top level BVH
            commandBuffer.buildAccelerationStructureNVX(
                vk::AccelerationStructureTypeNVX::eTopLevel,
                numInstancesNVX,     // instance count
                instancesNVX.buffer, // instancedata
                0,                   // instanceoffset
                0,                   // geomcount
                nullptr,             // geometries
                vk::BuildAccelerationStructureFlagBitsNVX::ePreferFastTrace, vk::Bool32(false), topHandle, nullptr,
                scratchMem.buffer,
                0,  // scratch offset
                loaderNVX
            );
        });
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 2 }, // 1 for vertex shader, 1 for raygen shader
            // Graphics pipeline:
            { vk::DescriptorType::eCombinedImageSampler, 4 },
            // Raytracing pipeline:
            { vk::DescriptorType::eStorageImage, 1 },
            { vk::DescriptorType::eAccelerationStructureNVX, 1 },
            { vk::DescriptorType::eStorageBuffer, 2 }, // hit shader: vertices, indices
        };

        descriptorPool = device.createDescriptorPool({ {}, /*maxSets*/3, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    // Prepare the raytracing pipeline
    void prepareRaytracing() {

        // Descriptor set layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Ray generation stage
            { 0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenNVX },
            { 1, vk::DescriptorType::eAccelerationStructureNVX, 1, vk::ShaderStageFlagBits::eRaygenNVX },
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eRaygenNVX }, // RT uniform buffer(camera params etc.)
            // Intersection stages
            { 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitNVX }, // indices
            { 4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitNVX }, // vertices
        };

        
        // Pipeline layout
        raytracingDescriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        raytracingPipelineLayout = device.createPipelineLayout({ {}, 1, &raytracingDescriptorSetLayout });
        
        // Pipeline
        std::array<vk::PipelineShaderStageCreateInfo, RT_STAGE_COUNT> rtStages;
        rtStages.at(0) = vks::shaders::loadShader(device,
            getAssetPath() + "shaders/nvx_raytracing/raygen.rgen.spv", vk::ShaderStageFlagBits::eRaygenNVX, "main");
        rtStages.at(1) = vks::shaders::loadShader(device,
            getAssetPath() + "shaders/nvx_raytracing/miss_primary.rmiss.spv", vk::ShaderStageFlagBits::eMissNVX, "main");
        rtStages.at(2) = vks::shaders::loadShader(device,
            getAssetPath() + "shaders/nvx_raytracing/diffuse.rchit.spv", vk::ShaderStageFlagBits::eClosestHitNVX, "main");

        const uint32_t groupNumbers[] = { 0, 1, 2 }; // [raygen, prim_miss, diffuse_hit]
        static_assert(std::size(groupNumbers) == RT_STAGE_COUNT, "Missing group numbers");
        auto createInfo = vk::RaytracingPipelineCreateInfoNVX({}, rtStages.size(), rtStages.data(), groupNumbers, 3/*rtDevMaxRecursionDepth*/, raytracingPipelineLayout);
        pipelines.raytracing = device.createRaytracingPipelineNVX(context.pipelineCache, createInfo, nullptr, loaderNVX);

        // Descriptor set
        auto sets = device.allocateDescriptorSets({ descriptorPool, 1, &raytracingDescriptorSetLayout });
        raytracingDescriptorSet = sets[0];
        updateRTDescriptorSets();
    }

    void updateRTDescriptorSets() {
        auto accelInfo = vk::DescriptorAccelerationStructureInfoNVX(1, &topHandle);
        std::vector<vk::DescriptorImageInfo> rtTexDescriptors{
            { nullptr, textureRaytracingTarget.view, vk::ImageLayout::eGeneral },
        };

        std::array<vk::WriteDescriptorSet, 5> rtWriteDescSets;
        rtWriteDescSets.at(0) = { raytracingDescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &rtTexDescriptors[0] };
        rtWriteDescSets.at(1) = { raytracingDescriptorSet, 1, 0, 1, vk::DescriptorType::eAccelerationStructureNVX }; rtWriteDescSets.at(1).pNext = &accelInfo;
        rtWriteDescSets.at(2) = { raytracingDescriptorSet, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataRaytracing.descriptor };
        rtWriteDescSets.at(3) = { raytracingDescriptorSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &meshes.rtMesh.indices.descriptor };
        rtWriteDescSets.at(4) = { raytracingDescriptorSet, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &meshes.rtMesh.vertices.descriptor };

        device.updateDescriptorSets(rtWriteDescSets, nullptr);
    }

    void setupShaderBindingTable() {
        const uint32_t bindingTableSize = RT_STAGE_COUNT * rtDevShaderHeaderSize;
        auto handles = device.getRaytracingShaderHandlesNVX(pipelines.raytracing, 0, RT_STAGE_COUNT, bindingTableSize, loaderNVX);
        shaderBindingTable = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eRaytracingNVX, handles);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformDataRaytracing = context.createUniformBuffer(uboRT);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboRT.lightPos.x = 0.0f + sin(glm::radians(timer * 360.0f)) * 2.0f;
        uboRT.lightPos.y = 5.0f;
        uboRT.lightPos.z = 1.0f;
        uboRT.lightPos.z = 0.0f + cos(glm::radians(timer * 360.0f)) * 2.0f;
        
        uboRT.camPos = glm::vec4(camera.position, 1.0f);
        glm::mat3 invR = glm::inverse(glm::mat3(camera.matrices.view));
        uboRT.invR = glm::mat4(invR);
        uboRT.aspectRatio = (float)size.width / (float)size.height;
        
        uniformDataRaytracing.copy(uboRT);
    }

    // Create a separate raytracing device queue
    void getRaytracingQueue() {
        uint32_t queueIndex = 0;

        std::vector<vk::QueueFamilyProperties> queueProps = physicalDevice.getQueueFamilyProperties();
        uint32_t queueCount = (uint32_t)queueProps.size();

        for (queueIndex = 0; queueIndex < queueCount; queueIndex++) {
            if (queueProps[queueIndex].queueFlags & vk::QueueFlagBits::eGraphics)
                break;
        }
        assert(queueIndex < queueCount);

        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.queueFamilyIndex = queueIndex;
        queueCreateInfo.queueCount = 1;
        raytracingQueue = device.getQueue(queueIndex, 0);
    }

    void prepare() {
        ExampleBase::prepare();
        getRTDeviceInfo();
        loaderNVX = vk::DispatchLoaderDynamic(context.instance, device);  // get NVX function pointers at runtime
        loadGeometry();
        getRaytracingQueue();
        buildAccelerationStructure();
        createRaytracingCommandBuffer();
        prepareUniformBuffers();
        prepareTextureTarget(textureRaytracingTarget, this->width, this->height, vk::Format::eR8G8B8A8Unorm);
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        prepareRaytracing();
        setupShaderBindingTable();
        buildCommandBuffers();
        updateRaytracingCommandBuffer();
        prepared = true;
    }

    virtual void windowResized() override {
        textureRaytracingTarget.destroy();
        prepareTextureTarget(textureRaytracingTarget, this->width, this->height, vk::Format::eR8G8B8A8Unorm);
        updateRaytracingCommandBuffer();
        updateDescriptorSet();
        updateRTDescriptorSets();
        uboRT.aspectRatio = (float)size.width / (float)size.height;
    }

    void mouseScrolled(float delta) override {
        float newSpeed = (delta > 0) ? camera.movementSpeed * 1.2f : camera.movementSpeed / 1.2f;
        camera.movementSpeed = std::max(1e-3f, std::min(1e6f, newSpeed));
    }

    virtual void render() override {
        if (!prepared)
            return;
        draw();
        traceRays();
        if (!paused) {
            updateUniformBuffers();
        }
    }

    virtual void viewChanged() { updateUniformBuffers(); }

    void initVulkan() override {
        context.requireExtensions({ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME });
        context.requireDeviceExtensions({ VK_NVX_RAYTRACING_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME });
        vkx::ExampleBase::initVulkan();
    }
};

RUN_EXAMPLE(VulkanExample)
