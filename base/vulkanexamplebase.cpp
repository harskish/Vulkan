/*
* Vulkan Example base class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"

void VulkanExampleBase::createInstance(bool enableValidation)
{
	this->enableValidation = enableValidation;

	vk::ApplicationInfo appInfo = {};
	appInfo.sType = vk::StructureType::eApplicationInfo;
	appInfo.pApplicationName = name.c_str();
	appInfo.pEngineName = name.c_str();
	appInfo.apiVersion = VK_API_VERSION_1_0;

	std::vector<const char*> enabledExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

	// Enable surface extensions depending on os
#if defined(_WIN32)
	enabledExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__ANDROID__)
	enabledExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(__linux__)
	enabledExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif

	vk::InstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = vk::StructureType::eInstanceCreateInfo;
	instanceCreateInfo.pNext = NULL;
	instanceCreateInfo.pApplicationInfo = &appInfo;
	if (enabledExtensions.size() > 0)
	{
		if (enableValidation)
		{
			enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}
		instanceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
	}
	if (enableValidation)
	{
		instanceCreateInfo.enabledLayerCount = vkDebug::validationLayerCount;
		instanceCreateInfo.ppEnabledLayerNames = vkDebug::validationLayerNames;
	}
	instance = vk::createInstance(instanceCreateInfo, nullptr);
}

void VulkanExampleBase::createDevice(vk::DeviceQueueCreateInfo requestedQueues, bool enableValidation)
{
	std::vector<const char*> enabledExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	vk::DeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = vk::StructureType::eDeviceCreateInfo;
	deviceCreateInfo.pNext = NULL;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &requestedQueues;
	deviceCreateInfo.pEnabledFeatures = NULL;

	// enable the debug marker extension if it is present (likely meaning a debugging tool is present)
	if (vkTools::checkDeviceExtensionPresent(physicalDevice, VK_EXT_DEBUG_MARKER_EXTENSION_NAME))
	{
		enabledExtensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
		enableDebugMarkers = true;
	}

	if (enabledExtensions.size() > 0)
	{
		deviceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
	}
	if (enableValidation)
	{
		deviceCreateInfo.enabledLayerCount = vkDebug::validationLayerCount;
		deviceCreateInfo.ppEnabledLayerNames = vkDebug::validationLayerNames;
	}

	device = physicalDevice.createDevice(deviceCreateInfo, nullptr);
}

std::string VulkanExampleBase::getWindowTitle()
{
	std::string device(deviceProperties.deviceName);
	std::string windowTitle;
	windowTitle = title + " - " + device + " - " + std::to_string(frameCounter) + " fps";
	return windowTitle;
}

const std::string VulkanExampleBase::getAssetPath()
{
#if defined(__ANDROID__)
	return "";
#else
	return "./../data/";
#endif
}

bool VulkanExampleBase::checkCommandBuffers()
{
	for (auto& cmdBuffer : drawCmdBuffers)
	{
		if (!cmdBuffer)
		{
			return false;
		}
	}
	return true;
}

void VulkanExampleBase::createCommandBuffers()
{
	// Create one command buffer per frame buffer
	// in the swap chain
	// Command buffers store a reference to the
	// frame buffer inside their render pass info
	// so for static usage withouth having to rebuild
	// them each frame, we use one per frame buffer
	vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
		vkTools::initializers::commandBufferAllocateInfo(
			cmdPool,
			vk::CommandBufferLevel::ePrimary,
			(uint32_t)swapChain.imageCount);

	drawCmdBuffers = device.allocateCommandBuffers(cmdBufAllocateInfo);

	// Command buffers for submitting present barriers
	cmdBufAllocateInfo.commandBufferCount = 1;
	// Pre present
	prePresentCmdBuffer = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];
	// Post present
	postPresentCmdBuffer = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];
}

void VulkanExampleBase::destroyCommandBuffers()
{
	device.freeCommandBuffers(cmdPool, (uint32_t)drawCmdBuffers.size(), drawCmdBuffers.data());
	device.freeCommandBuffers(cmdPool, prePresentCmdBuffer);
	device.freeCommandBuffers(cmdPool, postPresentCmdBuffer);
}

void VulkanExampleBase::createSetupCommandBuffer()
{
	if (setupCmdBuffer)
	{
		device.freeCommandBuffers(cmdPool, setupCmdBuffer);
		setupCmdBuffer; // todo : check if still necessary
	}

	vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
		vkTools::initializers::commandBufferAllocateInfo(
			cmdPool,
			vk::CommandBufferLevel::ePrimary,
			1);

	setupCmdBuffer = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];

	vk::CommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType = vk::StructureType::eCommandBufferBeginInfo;

	setupCmdBuffer.begin(cmdBufInfo);
}

void VulkanExampleBase::flushSetupCommandBuffer()
{
	if (!setupCmdBuffer)
		return;

	setupCmdBuffer.end();

	vk::SubmitInfo submitInfo = {};
	submitInfo.sType = vk::StructureType::eSubmitInfo;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &setupCmdBuffer;

	queue.submit(submitInfo, vk::Fence());
	queue.waitIdle();

	device.freeCommandBuffers(cmdPool, setupCmdBuffer);
	setupCmdBuffer; 
}

vk::CommandBuffer VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel level, bool begin)
{
	vk::CommandBuffer cmdBuffer;

	vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
		vkTools::initializers::commandBufferAllocateInfo(
			cmdPool,
			level,
			1);

	cmdBuffer = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];

	// If requested, also start the new command buffer
	if (begin)
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vkTools::initializers::commandBufferBeginInfo();
		cmdBuffer.begin(cmdBufInfo);
	}

	return cmdBuffer;
}

void VulkanExampleBase::flushCommandBuffer(vk::CommandBuffer commandBuffer, vk::Queue queue, bool free)
{
	if (!commandBuffer)
	{
		return;
	}
	
	commandBuffer.end();

	vk::SubmitInfo submitInfo = {};
	submitInfo.sType = vk::StructureType::eSubmitInfo;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	queue.submit(submitInfo, VK_NULL_HANDLE);
	queue.waitIdle();

	if (free)
	{
		device.freeCommandBuffers(cmdPool, commandBuffer);
	}
}

void VulkanExampleBase::createPipelineCache()
{
	vk::PipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = vk::StructureType::ePipelineCacheCreateInfo;
	pipelineCache = device.createPipelineCache(pipelineCacheCreateInfo, nullptr);
}

void VulkanExampleBase::prepare()
{
	if (enableValidation)
	{
		vkDebug::setupDebugging(instance, vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning, VK_NULL_HANDLE);
	}
	if (enableDebugMarkers)
	{
		vkDebug::DebugMarker::setup(device);
	}
	createCommandPool();
	createSetupCommandBuffer();
	setupSwapChain();
	createCommandBuffers();
	setupDepthStencil();
	setupRenderPass();
	createPipelineCache();
	setupFrameBuffer();
	flushSetupCommandBuffer();
	// Recreate setup command buffer for derived class
	createSetupCommandBuffer();
	// Create a simple texture loader class
	textureLoader = new vkTools::VulkanTextureLoader(physicalDevice, device, queue, cmdPool);
#if defined(__ANDROID__)
	textureLoader->assetManager = androidApp->activity->assetManager;
#endif
	if (enableTextOverlay)
	{
		// Load the text rendering shaders
		std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
		shaderStages.push_back(loadShader(getAssetPath() + "shaders/base/textoverlay.vert.spv", vk::ShaderStageFlagBits::eVertex));
		shaderStages.push_back(loadShader(getAssetPath() + "shaders/base/textoverlay.frag.spv", vk::ShaderStageFlagBits::eFragment));
		textOverlay = new VulkanTextOverlay(
			physicalDevice,
			device,
			queue,
			frameBuffers,
			colorformat,
			depthFormat,
			&width,
			&height,
			shaderStages
			);
		updateTextOverlay();
	}
}

vk::PipelineShaderStageCreateInfo VulkanExampleBase::loadShader(std::string fileName, vk::ShaderStageFlagBits stage)
{
	vk::PipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
	shaderStage.stage = stage;
#if defined(__ANDROID__)
	shaderStage.module = vkTools::loadShader(androidApp->activity->assetManager, fileName.c_str(), device, stage);
#else
	shaderStage.module = vkTools::loadShader(fileName.c_str(), device, stage);
#endif
	shaderStage.pName = "main"; // todo : make param
	assert(shaderStage.module);
	shaderModules.push_back(shaderStage.module);
	return shaderStage;
}

vk::Bool32 VulkanExampleBase::createBuffer(vk::BufferUsageFlags usageFlags, vk::MemoryPropertyFlags memoryPropertyFlags, vk::DeviceSize size, void * data, vk::Buffer & buffer, vk::DeviceMemory & memory)
{
	vk::MemoryRequirements memReqs;
	vk::MemoryAllocateInfo memAlloc = vkTools::initializers::memoryAllocateInfo();
	vk::BufferCreateInfo bufferCreateInfo = vkTools::initializers::bufferCreateInfo(usageFlags, size);

	buffer = device.createBuffer(bufferCreateInfo, nullptr);

	memReqs = device.getBufferMemoryRequirements(buffer);
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
	memory = device.allocateMemory(memAlloc, nullptr);
	if (data != nullptr)
	{
		void *mapped;
		mapped = device.mapMemory(memory, 0, size, vk::MemoryMapFlags());
		memcpy(mapped, data, size);
		device.unmapMemory(memory);
	}
	device.bindBufferMemory(buffer, memory, 0);

	return true;
}

vk::Bool32 VulkanExampleBase::createBuffer(vk::BufferUsageFlags usage, vk::DeviceSize size, void * data, vk::Buffer &buffer, vk::DeviceMemory &memory)
{
	return createBuffer(usage, vk::MemoryPropertyFlagBits::eHostVisible, size, data, buffer, memory);
}

vk::Bool32 VulkanExampleBase::createBuffer(vk::BufferUsageFlags usage, vk::DeviceSize size, void * data, vk::Buffer & buffer, vk::DeviceMemory & memory, vk::DescriptorBufferInfo & descriptor)
{
	vk::Bool32 res = createBuffer(usage, size, data, buffer, memory);
	if (res)
	{
		descriptor.offset = 0;
		descriptor.buffer = buffer;
		descriptor.range = size;
		return true;
	}
	else
	{
		return false;
	}
}

vk::Bool32 VulkanExampleBase::createBuffer(vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryPropertyFlags, vk::DeviceSize size, void * data, vk::Buffer & buffer, vk::DeviceMemory& memory, vk::DescriptorBufferInfo & descriptor)
{
	vk::Bool32 res = createBuffer(usage, memoryPropertyFlags, size, data, buffer, memory);
	if (res)
	{
		descriptor.offset = 0;
		descriptor.buffer = buffer;
		descriptor.range = size;
		return true;
	}
	else
	{
		return false;
	}
}

void VulkanExampleBase::loadMesh(
	std::string filename,
	vkMeshLoader::MeshBuffer * meshBuffer,
	std::vector<vkMeshLoader::VertexLayout> vertexLayout,
	float scale)
{
	VulkanMeshLoader *mesh = new VulkanMeshLoader();

#if defined(__ANDROID__)
	mesh->assetManager = androidApp->activity->assetManager;
#endif

	mesh->LoadMesh(filename);
	assert(mesh->m_Entries.size() > 0);

	vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);

	mesh->createBuffers(
		device,
		deviceMemoryProperties,
		meshBuffer,
		vertexLayout,
		scale,
		true,
		copyCmd,
		queue);

	device.freeCommandBuffers(cmdPool, copyCmd);

	meshBuffer->dim = mesh->dim.size;

	delete(mesh);
}

void VulkanExampleBase::renderLoop()
{
	destWidth = width;
	destHeight = height;
#if defined(_WIN32)
	MSG msg;
	while (TRUE)
	{
		auto tStart = std::chrono::high_resolution_clock::now();
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				break;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		render();
		frameCounter++;
		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		frameTimer = (float)tDiff / 1000.0f;
		// Convert to clamped timer value
		if (!paused)
		{
			timer += timerSpeed * frameTimer;
			if (timer > 1.0)
			{
				timer -= 1.0f;
			}
		}
		fpsTimer += (float)tDiff;
		if (fpsTimer > 1000.0f)
		{
			std::string windowTitle = getWindowTitle();
			if (!enableTextOverlay)
			{
				SetWindowText(window, windowTitle.c_str());
			}
			lastFPS = frameCounter;
			updateTextOverlay();
			fpsTimer = 0.0f;
			frameCounter = 0;
		}
	}
#elif defined(__ANDROID__)
	while (1)
	{
		int ident;
		int events;
		struct android_poll_source* source;
		bool destroy = false;

		focused = true;

		while ((ident = ALooper_pollAll(focused ? 0 : -1, NULL, &events, (void**)&source)) >= 0)
		{
			if (source != NULL)
			{
				source->process(androidApp, source);
			}
			if (androidApp->destroyRequested != 0)
			{
				LOGD("Android app destroy requested");
				destroy = true;
				break;
			}
		}

		// App destruction requested
		// Exit loop, example will be destroyed in application main
		if (destroy)
		{
			break;
		}

		// Render frame
		if (prepared)
		{
			auto tStart = std::chrono::high_resolution_clock::now();
			render();
			frameCounter++;
			auto tEnd = std::chrono::high_resolution_clock::now();
			auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
			frameTimer = tDiff / 1000.0f;
			// Convert to clamped timer value
			if (!paused)
			{
				timer += timerSpeed * frameTimer;
				if (timer > 1.0)
				{
					timer -= 1.0f;
				}
			}
			fpsTimer += (float)tDiff;
			if (fpsTimer > 1000.0f)
			{
				lastFPS = frameCounter;
				updateTextOverlay();
				fpsTimer = 0.0f;
				frameCounter = 0;
			}
			// Check gamepad state
			const float deadZone = 0.0015f;
			// todo : check if gamepad is present
			// todo : time based and relative axis positions
			bool updateView = false;
			// Rotate
			if (std::abs(gamePadState.axes.x) > deadZone)
			{
				rotation.y += gamePadState.axes.x * 0.5f * rotationSpeed;
				updateView = true;
			}
			if (std::abs(gamePadState.axes.y) > deadZone)
			{
				rotation.x -= gamePadState.axes.y * 0.5f * rotationSpeed;
				updateView = true;
			}
			// Zoom
			if (std::abs(gamePadState.axes.rz) > deadZone)
			{
				zoom -= gamePadState.axes.rz * 0.01f * zoomSpeed;
				updateView = true;
			}
			if (updateView)
			{
				viewChanged();
			}


		}
	}
#elif defined(__linux__)
	xcb_flush(connection);
	while (!quit)
	{
		auto tStart = std::chrono::high_resolution_clock::now();
		xcb_generic_event_t *event;
		event = xcb_poll_for_event(connection);
		if (event)
		{
			handleEvent(event);
			free(event);
		}
		render();
		frameCounter++;
		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		frameTimer = tDiff / 1000.0f;
		// Convert to clamped timer value
		if (!paused)
		{
			timer += timerSpeed * frameTimer;
			if (timer > 1.0)
			{
				timer -= 1.0f;
			}
		}
		fpsTimer += (float)tDiff;
		if (fpsTimer > 1000.0f)
		{
			if (!enableTextOverlay)
			{
				std::string windowTitle = getWindowTitle();
				xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
					window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
					windowTitle.size(), windowTitle.c_str());
			}
			lastFPS = frameCounter;
			updateTextOverlay();
			fpsTimer = 0.0f;
			frameCounter = 0;
		}
	}
#endif
}

void VulkanExampleBase::submitPrePresentBarrier(vk::Image image)
{
	vk::CommandBufferBeginInfo cmdBufInfo = vkTools::initializers::commandBufferBeginInfo();

	prePresentCmdBuffer.begin(cmdBufInfo);

	vk::ImageMemoryBarrier prePresentBarrier = vkTools::initializers::imageMemoryBarrier();
	prePresentBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
	prePresentBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
	prePresentBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
	prePresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	prePresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	prePresentBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
	prePresentBarrier.image = image;

	prePresentCmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTopOfPipe, vk::DependencyFlags(), nullptr, nullptr, prePresentBarrier);

	prePresentCmdBuffer.end();

	vk::SubmitInfo submitInfo = vkTools::initializers::submitInfo();
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &prePresentCmdBuffer;

	queue.submit(submitInfo, VK_NULL_HANDLE);
}

void VulkanExampleBase::submitPostPresentBarrier(vk::Image image)
{
	vk::CommandBufferBeginInfo cmdBufInfo = vkTools::initializers::commandBufferBeginInfo();

	postPresentCmdBuffer.begin(cmdBufInfo);

	vk::ImageMemoryBarrier postPresentBarrier = vkTools::initializers::imageMemoryBarrier();
	postPresentBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
	postPresentBarrier.oldLayout = vk::ImageLayout::ePresentSrcKHR;
	postPresentBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
	postPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	postPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	postPresentBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
	postPresentBarrier.image = image;

	postPresentCmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTopOfPipe, vk::DependencyFlags(), nullptr, nullptr, postPresentBarrier);

	postPresentCmdBuffer.end();

	vk::SubmitInfo submitInfo = vkTools::initializers::submitInfo();
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &postPresentCmdBuffer;

	queue.submit(submitInfo, VK_NULL_HANDLE);
}

vk::SubmitInfo VulkanExampleBase::prepareSubmitInfo(
	std::vector<vk::CommandBuffer> commandBuffers,
	vk::PipelineStageFlags *pipelineStages)
{
	vk::SubmitInfo submitInfo = vkTools::initializers::submitInfo();
	submitInfo.pWaitDstStageMask = pipelineStages;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &semaphores.presentComplete;
	submitInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
	submitInfo.pCommandBuffers = commandBuffers.data();
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &semaphores.renderComplete;
	return submitInfo;
}

void VulkanExampleBase::updateTextOverlay()
{
	if (!enableTextOverlay)
		return;

	textOverlay->beginTextUpdate();

	textOverlay->addText(title, 5.0f, 5.0f, VulkanTextOverlay::alignLeft);

	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << (frameTimer * 1000.0f) << "ms (" << lastFPS << " fps)";
	textOverlay->addText(ss.str(), 5.0f, 25.0f, VulkanTextOverlay::alignLeft);

	textOverlay->addText(deviceProperties.deviceName, 5.0f, 45.0f, VulkanTextOverlay::alignLeft);

	getOverlayText(textOverlay);

	textOverlay->endTextUpdate();
}

void VulkanExampleBase::getOverlayText(VulkanTextOverlay *textOverlay)
{
	// Can be overriden in derived class
}

void VulkanExampleBase::prepareFrame()
{
	// Acquire the next image from the swap chaing
	swapChain.acquireNextImage(semaphores.presentComplete, currentBuffer);
	// Submit barrier that transforms color attachment image layout back from khr
	submitPostPresentBarrier(swapChain.buffers[currentBuffer].image);

}

void VulkanExampleBase::submitFrame()
{
	bool submitTextOverlay = enableTextOverlay && textOverlay->visible;

	if (submitTextOverlay)
	{
		// Wait for color attachment output to finish before rendering the text overlay
		vk::PipelineStageFlags stageFlags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		submitInfo.pWaitDstStageMask = &stageFlags;

		// Set semaphores
		// Wait for render complete semaphore
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &semaphores.renderComplete;
		// Signal ready with text overlay complete semaphpre
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &semaphores.textOverlayComplete;

		// Submit current text overlay command buffer
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &textOverlay->cmdBuffers[currentBuffer];
		queue.submit(submitInfo, VK_NULL_HANDLE);

		// Reset stage mask
		submitInfo.pWaitDstStageMask = &submitPipelineStages;
		// Reset wait and signal semaphores for rendering next frame
		// Wait for swap chain presentation to finish
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		// Signal ready with offscreen semaphore
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;
	}

	// Submit barrier that transforms color attachment to khr presen
	submitPrePresentBarrier(swapChain.buffers[currentBuffer].image);

	swapChain.queuePresent(queue, currentBuffer, submitTextOverlay ? semaphores.textOverlayComplete : semaphores.renderComplete);

	queue.waitIdle();
}

VulkanExampleBase::VulkanExampleBase(bool enableValidation)
{
	// Check for validation command line flag
#if defined(_WIN32)
	for (int32_t i = 0; i < __argc; i++)
	{
		if (__argv[i] == std::string("-validation"))
		{
			enableValidation = true;
		}
	}
#elif defined(__ANDROID__)
	// Vulkan library is loaded dynamically on Android
	bool libLoaded = loadVulkanLibrary();
	assert(libLoaded);
#elif defined(__linux__)
	initxcbConnection();
#endif

#if !defined(__ANDROID__)
	// Android Vulkan initialization is handled in APP_CMD_INIT_WINDOW event
	initVulkan(enableValidation);
#endif

#if defined(_WIN32)
	// Enable console if validation is active
	// Debug message callback will output to it
	if (enableValidation)
	{
		setupConsole("VulkanExample");
	}
#endif
}

VulkanExampleBase::~VulkanExampleBase()
{
	// Clean up Vulkan resources
	swapChain.cleanup();
	if (descriptorPool)
	{
		device.destroyDescriptorPool(descriptorPool, nullptr);
	}
	if (setupCmdBuffer)
	{
		device.freeCommandBuffers(cmdPool, setupCmdBuffer);

	}
	destroyCommandBuffers();
	device.destroyRenderPass(renderPass, nullptr);
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		device.destroyFramebuffer(frameBuffers[i], nullptr);
	}

	for (auto& shaderModule : shaderModules)
	{
		device.destroyShaderModule(shaderModule, nullptr);
	}
	device.destroyImageView(depthStencil.view, nullptr);
	device.destroyImage(depthStencil.image, nullptr);
	device.freeMemory(depthStencil.mem, nullptr);

	device.destroyPipelineCache(pipelineCache, nullptr);

	if (textureLoader)
	{
		delete textureLoader;
	}

	device.destroyCommandPool(cmdPool, nullptr);

	device.destroySemaphore(semaphores.presentComplete, nullptr);
	device.destroySemaphore(semaphores.renderComplete, nullptr);
	device.destroySemaphore(semaphores.textOverlayComplete, nullptr);

	if (enableTextOverlay)
	{
		delete textOverlay;
	}

	device.destroy();

	if (enableValidation)
	{
		vkDebug::freeDebugCallback(instance);
	}

	instance.destroy();

#if defined(__linux)
#if defined(__ANDROID__)
	// todo : android cleanup (if required)
#else
	xcb_destroy_window(connection, window);
	xcb_disconnect(connection);
#endif
#endif
}

void VulkanExampleBase::initVulkan(bool enableValidation)
{
	// Vulkan instance
	createInstance(enableValidation);
	//if (err)
	//{
	//	vkTools::exitFatal("Could not create Vulkan instance : \n" + vkTools::errorString(err), "Fatal error");
	//}

#if defined(__ANDROID__)
	loadVulkanFunctions(instance);
#endif

	// Physical device
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

	// Note :
	// This example will always use the first physical device reported,
	// change the vector index if you have multiple Vulkan devices installed
	// and want to use another one
	physicalDevice = physicalDevices[0];

	// Find a queue that supports graphics operations
	uint32_t graphicsQueueIndex = 0;
	std::vector<vk::QueueFamilyProperties> queueProps = physicalDevice.getQueueFamilyProperties();;
	uint32_t queueCount = queueProps.size();

	for (graphicsQueueIndex = 0; graphicsQueueIndex < queueCount; graphicsQueueIndex++)
	{
		if (queueProps[graphicsQueueIndex].queueFlags & vk::QueueFlagBits::eGraphics)
			break;
	}
	assert(graphicsQueueIndex < queueCount);

	// Vulkan device
	std::array<float, 1> queuePriorities = { 0.0f };
	vk::DeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = vk::StructureType::eDeviceQueueCreateInfo;
	queueCreateInfo.queueFamilyIndex = graphicsQueueIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = queuePriorities.data();

	createDevice(queueCreateInfo, enableValidation);

	// Store properties (including limits) and features of the phyiscal device
	// So examples can check against them and see if a feature is actually supported
	deviceProperties = physicalDevice.getProperties();
	deviceFeatures = physicalDevice.getFeatures();

	// Gather physical device memory properties
	deviceMemoryProperties = physicalDevice.getMemoryProperties();

	// Get the graphics queue
	queue = device.getQueue(graphicsQueueIndex, 0);

	// Find a suitable depth format
	vk::Bool32 validDepthFormat = vkTools::getSupportedDepthFormat(physicalDevice, &depthFormat);
	assert(validDepthFormat);

	swapChain.connect(instance, physicalDevice, device);

	// Create synchronization objects
	vk::SemaphoreCreateInfo semaphoreCreateInfo = vkTools::initializers::semaphoreCreateInfo();
	// Create a semaphore used to synchronize image presentation
	// Ensures that the image is displayed before we start submitting new commands to the queu
	semaphores.presentComplete = device.createSemaphore(semaphoreCreateInfo, nullptr);
	// Create a semaphore used to synchronize command submission
	// Ensures that the image is not presented until all commands have been sumbitted and executed
	semaphores.renderComplete = device.createSemaphore(semaphoreCreateInfo, nullptr);
	// Create a semaphore used to synchronize command submission
	// Ensures that the image is not presented until all commands for the text overlay have been sumbitted and executed
	// Will be inserted after the render complete semaphore if the text overlay is enabled
	semaphores.textOverlayComplete = device.createSemaphore(semaphoreCreateInfo, nullptr);

	// Set up submit info structure
	// Semaphores will stay the same during application lifetime
	// Command buffer submission info is set by each example
	submitInfo = vkTools::initializers::submitInfo();
	submitInfo.pWaitDstStageMask = &submitPipelineStages;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &semaphores.presentComplete;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &semaphores.renderComplete;
}

#if defined(_WIN32)
// Win32 : Sets up a console window and redirects standard output to it
void VulkanExampleBase::setupConsole(std::string title)
{
	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	FILE *stream;
	freopen_s(&stream, "CONOUT$", "w+", stdout);
	SetConsoleTitle(TEXT(title.c_str()));
	if (enableValidation)
	{
		std::cout << "Validation enabled:\n";
	}
}

HWND VulkanExampleBase::setupWindow(HINSTANCE hinstance, WNDPROC wndproc)
{
	this->windowInstance = hinstance;

	bool fullscreen = false;

	// Check command line arguments
	for (int32_t i = 0; i < __argc; i++)
	{
		if (__argv[i] == std::string("-fullscreen"))
		{
			fullscreen = true;
		}
	}

	WNDCLASSEX wndClass;

	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.style = CS_HREDRAW | CS_VREDRAW;
	wndClass.lpfnWndProc = wndproc;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hinstance;
	wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndClass.lpszMenuName = NULL;
	wndClass.lpszClassName = name.c_str();
	wndClass.hIconSm = LoadIcon(NULL, IDI_WINLOGO);

	if (!RegisterClassEx(&wndClass))
	{
		std::cout << "Could not register window class!\n";
		fflush(stdout);
		exit(1);
	}

	int screenWidth = GetSystemMetrics(SM_CXSCREEN);
	int screenHeight = GetSystemMetrics(SM_CYSCREEN);

	if (fullscreen)
	{
		DEVMODE dmScreenSettings;
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth = screenWidth;
		dmScreenSettings.dmPelsHeight = screenHeight;
		dmScreenSettings.dmBitsPerPel = 32;
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		if ((width != screenWidth) && (height != screenHeight))
		{
			if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL)
			{
				if (MessageBox(NULL, "Fullscreen Mode not supported!\n Switch to window mode?", "Error", MB_YESNO | MB_ICONEXCLAMATION) == IDYES)
				{
					fullscreen = FALSE;
				}
				else
				{
					return FALSE;
				}
			}
		}

	}

	DWORD dwExStyle;
	DWORD dwStyle;

	if (fullscreen)
	{
		dwExStyle = WS_EX_APPWINDOW;
		dwStyle = WS_POPUP | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
	}
	else
	{
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
		dwStyle = WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
	}

	RECT windowRect;
	if (fullscreen)
	{
		windowRect.left = (long)0;
		windowRect.right = (long)screenWidth;
		windowRect.top = (long)0;
		windowRect.bottom = (long)screenHeight;
	}
	else
	{
		windowRect.left = (long)screenWidth / 2 - width / 2;
		windowRect.right = (long)width;
		windowRect.top = (long)screenHeight / 2 - height / 2;
		windowRect.bottom = (long)height;
	}

	AdjustWindowRectEx(&windowRect, dwStyle, FALSE, dwExStyle);

	std::string windowTitle = getWindowTitle();
	window = CreateWindowEx(0,
		name.c_str(),
		windowTitle.c_str(),
		dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		windowRect.left,
		windowRect.top,
		windowRect.right,
		windowRect.bottom,
		NULL,
		NULL,
		hinstance,
		NULL);

	if (!window)
	{
		printf("Could not create window!\n");
		fflush(stdout);
		return 0;
		exit(1);
	}

	ShowWindow(window, SW_SHOW);
	SetForegroundWindow(window);
	SetFocus(window);

	return window;
}

void VulkanExampleBase::handleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_CLOSE:
		prepared = false;
		DestroyWindow(hWnd);
		PostQuitMessage(0);
		break;
	case WM_PAINT:
		ValidateRect(window, NULL);
		break;
	case WM_KEYDOWN:
		switch (wParam)
		{
		case 0x50:
			paused = !paused;
			break;
		case VK_F1:
			if (enableTextOverlay)
			{
				textOverlay->visible = !textOverlay->visible;
			}
			break;
		case VK_ESCAPE:
			PostQuitMessage(0);
			break;
		}
		keyPressed((uint32_t)wParam);
		break;
	case WM_RBUTTONDOWN:
	case WM_LBUTTONDOWN:
	case WM_MBUTTONDOWN:
		mousePos.x = (float)LOWORD(lParam);
		mousePos.y = (float)HIWORD(lParam);
		break;
	case WM_MOUSEWHEEL:
	{
		short wheelDelta = GET_WHEEL_DELTA_WPARAM(wParam);
		zoom += (float)wheelDelta * 0.005f * zoomSpeed;
		viewChanged();
		break;
	}
	case WM_MOUSEMOVE:
		if (wParam & MK_RBUTTON)
		{
			int32_t posx = LOWORD(lParam);
			int32_t posy = HIWORD(lParam);
			zoom += (mousePos.y - (float)posy) * .005f * zoomSpeed;
			mousePos = glm::vec2((float)posx, (float)posy);
			viewChanged();
		}
		if (wParam & MK_LBUTTON)
		{
			int32_t posx = LOWORD(lParam);
			int32_t posy = HIWORD(lParam);
			rotation.x += (mousePos.y - (float)posy) * 1.25f * rotationSpeed;
			rotation.y -= (mousePos.x - (float)posx) * 1.25f * rotationSpeed;
			mousePos = glm::vec2((float)posx, (float)posy);
			viewChanged();
		}
		if (wParam & MK_MBUTTON)
		{
			int32_t posx = LOWORD(lParam);
			int32_t posy = HIWORD(lParam);
			cameraPos.x -= (mousePos.x - (float)posx) * 0.01f;
			cameraPos.y -= (mousePos.y - (float)posy) * 0.01f;
			viewChanged();
			mousePos.x = (float)posx;
			mousePos.y = (float)posy;
		}
		break;
	case WM_SIZE:
		if ((prepared) && (wParam != SIZE_MINIMIZED))
		{
			destWidth = LOWORD(lParam);
			destHeight = HIWORD(lParam);
			if ((wParam == SIZE_MAXIMIZED) || (wParam == SIZE_MINIMIZED))
			{
				windowResize();
			}
		}
		break;
	case WM_EXITSIZEMOVE:
		if ((prepared) && ((destWidth != width) || (destHeight != height)))
		{
			windowResize();
		}
		break;
	}
}
#elif defined(__ANDROID__)
int32_t VulkanExampleBase::handleAppInput(struct android_app* app, AInputEvent* event)
{
	VulkanExampleBase* vulkanExample = (VulkanExampleBase*)app->userData;
	if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
	{
		if (AInputEvent_getSource(event) == AINPUT_SOURCE_JOYSTICK)
		{
			vulkanExample->gamePadState.axes.x = AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_X, 0);
			vulkanExample->gamePadState.axes.y = AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_Y, 0);
			vulkanExample->gamePadState.axes.z = AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_Z, 0);
			vulkanExample->gamePadState.axes.rz = AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_RZ, 0);
		}
		else
		{
			// todo : touch input
		}
		return 1;
	}

	if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY)
	{
		int32_t keyCode = AKeyEvent_getKeyCode((const AInputEvent*)event);
		int32_t action = AKeyEvent_getAction((const AInputEvent*)event);
		int32_t button = 0;

		if (action == AKEY_EVENT_ACTION_UP)
			return 0;

		switch (keyCode)
		{
		case AKEYCODE_BUTTON_A:
			vulkanExample->keyPressed(GAMEPAD_BUTTON_A);
			break;
		case AKEYCODE_BUTTON_B:
			vulkanExample->keyPressed(GAMEPAD_BUTTON_B);
			break;
		case AKEYCODE_BUTTON_X:
			vulkanExample->keyPressed(GAMEPAD_BUTTON_X);
			break;
		case AKEYCODE_BUTTON_Y:
			vulkanExample->keyPressed(GAMEPAD_BUTTON_Y);
			break;
		case AKEYCODE_BUTTON_L1:
			vulkanExample->keyPressed(GAMEPAD_BUTTON_L1);
			break;
		case AKEYCODE_BUTTON_R1:
			vulkanExample->keyPressed(GAMEPAD_BUTTON_R1);
			break;
		case AKEYCODE_BUTTON_START:
			vulkanExample->keyPressed(GAMEPAD_BUTTON_START);
			break;
		};

		LOGD("Button %d pressed", keyCode);
	}

	return 0;
}

void VulkanExampleBase::handleAppCommand(android_app * app, int32_t cmd)
{
	assert(app->userData != NULL);
	VulkanExampleBase* vulkanExample = (VulkanExampleBase*)app->userData;
	switch (cmd)
	{
	case APP_CMD_SAVE_STATE:
		LOGD("APP_CMD_SAVE_STATE");
		/*
		vulkanExample->app->savedState = malloc(sizeof(struct saved_state));
		*((struct saved_state*)vulkanExample->app->savedState) = vulkanExample->state;
		vulkanExample->app->savedStateSize = sizeof(struct saved_state);
		*/
		break;
	case APP_CMD_INIT_WINDOW:
		LOGD("APP_CMD_INIT_WINDOW");
		if (vulkanExample->androidApp->window != NULL)
		{
			vulkanExample->initVulkan(false);
			vulkanExample->initSwapchain();
			vulkanExample->prepare();
			assert(vulkanExample->prepared);
		}
		else
		{
			LOGE("No window assigned!");
		}
		break;
	case APP_CMD_LOST_FOCUS:
		LOGD("APP_CMD_LOST_FOCUS");
		vulkanExample->focused = false;
		break;
	case APP_CMD_GAINED_FOCUS:
		LOGD("APP_CMD_GAINED_FOCUS");
		vulkanExample->focused = true;
		break;
	}
}
#elif defined(__linux__)
// Set up a window using XCB and request event types
xcb_window_t VulkanExampleBase::setupWindow()
{
	uint32_t value_mask, value_list[32];

	window = xcb_generate_id(connection);

	value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
	value_list[0] = screen->black_pixel;
	value_list[1] =
		XCB_EVENT_MASK_KEY_RELEASE |
		XCB_EVENT_MASK_EXPOSURE |
		XCB_EVENT_MASK_STRUCTURE_NOTIFY |
		XCB_EVENT_MASK_POINTER_MOTION |
		XCB_EVENT_MASK_BUTTON_PRESS |
		XCB_EVENT_MASK_BUTTON_RELEASE;

	xcb_create_window(connection,
		XCB_COPY_FROM_PARENT,
		window, screen->root,
		0, 0, width, height, 0,
		XCB_WINDOW_CLASS_INPUT_OUTPUT,
		screen->root_visual,
		value_mask, value_list);

	/* Magic code that will send notification when window is destroyed */
	xcb_intern_atom_cookie_t cookie = xcb_intern_atom(connection, 1, 12, "WM_PROTOCOLS");
	xcb_intern_atom_reply_t* reply = xcb_intern_atom_reply(connection, cookie, 0);

	xcb_intern_atom_cookie_t cookie2 = xcb_intern_atom(connection, 0, 16, "WM_DELETE_WINDOW");
	atom_wm_delete_window = xcb_intern_atom_reply(connection, cookie2, 0);

	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
		window, (*reply).atom, 4, 32, 1,
		&(*atom_wm_delete_window).atom);

	std::string windowTitle = getWindowTitle();
	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
		window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
		title.size(), windowTitle.c_str());

	free(reply);

	xcb_map_window(connection, window);

	return(window);
}

// Initialize XCB connection
void VulkanExampleBase::initxcbConnection()
{
	const xcb_setup_t *setup;
	xcb_screen_iterator_t iter;
	int scr;

	connection = xcb_connect(NULL, &scr);
	if (connection == NULL) {
		printf("Could not find a compatible Vulkan ICD!\n");
		fflush(stdout);
		exit(1);
	}

	setup = xcb_get_setup(connection);
	iter = xcb_setup_roots_iterator(setup);
	while (scr-- > 0)
		xcb_screen_next(&iter);
	screen = iter.data;
}

void VulkanExampleBase::handleEvent(const xcb_generic_event_t *event)
{
	switch (event->response_type & 0x7f)
	{
	case XCB_CLIENT_MESSAGE:
		if ((*(xcb_client_message_event_t*)event).data.data32[0] ==
			(*atom_wm_delete_window).atom) {
			quit = true;
		}
		break;
	case XCB_MOTION_NOTIFY:
	{
		xcb_motion_notify_event_t *motion = (xcb_motion_notify_event_t *)event;
		if (mouseButtons.left)
		{
			rotation.x += (mousePos.y - (float)motion->event_y) * 1.25f;
			rotation.y -= (mousePos.x - (float)motion->event_x) * 1.25f;
			viewChanged();
		}
		if (mouseButtons.right)
		{
			zoom += (mousePos.y - (float)motion->event_y) * .005f;
			viewChanged();
		}
		if (mouseButtons.middle)
		{
			cameraPos.x -= (mousePos.x - (float)motion->event_x) * 0.01f;
			cameraPos.y -= (mousePos.y - (float)motion->event_y) * 0.01f;
			viewChanged();
			mousePos.x = (float)motion->event_x;
			mousePos.y = (float)motion->event_y;
		}
		mousePos = glm::vec2((float)motion->event_x, (float)motion->event_y);
	}
	break;
	case XCB_BUTTON_PRESS:
	{
		xcb_button_press_event_t *press = (xcb_button_press_event_t *)event;
		if (press->detail == XCB_BUTTON_INDEX_1)
			mouseButtons.left = true;
		if (press->detail == XCB_BUTTON_INDEX_2)
			mouseButtons.middle = true;
		if (press->detail == XCB_BUTTON_INDEX_3)
			mouseButtons.right = true;
	}
	break;
	case XCB_BUTTON_RELEASE:
	{
		xcb_button_press_event_t *press = (xcb_button_press_event_t *)event;
		if (press->detail == XCB_BUTTON_INDEX_1)
			mouseButtons.left = false;
		if (press->detail == XCB_BUTTON_INDEX_2)
			mouseButtons.middle = false;
		if (press->detail == XCB_BUTTON_INDEX_3)
			mouseButtons.right = false;
	}
	break;
	case XCB_KEY_RELEASE:
	{
		const xcb_key_release_event_t *keyEvent = (const xcb_key_release_event_t *)event;
		if (keyEvent->detail == 0x9)
			quit = true;
		keyPressed(keyEvent->detail);
	}
	break;
	case XCB_DESTROY_NOTIFY:
		quit = true;
		break;
	case XCB_CONFIGURE_NOTIFY:
	{
		const xcb_configure_notify_event_t *cfgEvent = (const xcb_configure_notify_event_t *)event;
		if ((prepared) && ((cfgEvent->width != width) || (cfgEvent->height != height)))
		{
				destWidth = cfgEvent->width;
				destHeight = cfgEvent->height;
				if ((destWidth > 0) && (destHeight > 0))
				{
					windowResize();
				}
		}
	}
	break;
	default:
		break;
	}
}
#endif

void VulkanExampleBase::viewChanged()
{
	// Can be overrdiden in derived class
}

void VulkanExampleBase::keyPressed(uint32_t keyCode)
{
	// Can be overriden in derived class
}

void VulkanExampleBase::buildCommandBuffers()
{
	// Can be overriden in derived class
}

vk::Bool32 VulkanExampleBase::getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags properties, uint32_t * typeIndex)
{
	for (uint32_t i = 0; i < 32; i++)
	{
		if ((typeBits & 1) == 1)
		{
			if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				*typeIndex = i;
				return true;
			}
		}
		typeBits >>= 1;
	}
	return false;
}

uint32_t VulkanExampleBase::getMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags properties)
{
	uint32_t result = 0;
	if (!getMemoryType(typeBits, properties, &result)) {
		// todo : throw error
	}
	return result;
}

void VulkanExampleBase::createCommandPool()
{
	vk::CommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = vk::StructureType::eCommandPoolCreateInfo;
	cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
	cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	cmdPool = device.createCommandPool(cmdPoolInfo, nullptr);
}

void VulkanExampleBase::setupDepthStencil()
{
	vk::ImageCreateInfo image = {};
	image.imageType = vk::ImageType::e2D;
	image.format = depthFormat;
	image.extent = { width, height, 1 };
	image.mipLevels = 1;
	image.arrayLayers = 1;
	image.samples = vk::SampleCountFlagBits::e1;
	image.tiling = vk::ImageTiling::eOptimal;
	image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc;

	vk::MemoryAllocateInfo mem_alloc = {};
	mem_alloc.allocationSize = 0;
	mem_alloc.memoryTypeIndex = 0;

	vk::ImageViewCreateInfo depthStencilView = {};
	depthStencilView.viewType = vk::ImageViewType::e2D;
	depthStencilView.format = depthFormat;
	depthStencilView.subresourceRange = {};
	depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
	depthStencilView.subresourceRange.baseMipLevel = 0;
	depthStencilView.subresourceRange.levelCount = 1;
	depthStencilView.subresourceRange.baseArrayLayer = 0;
	depthStencilView.subresourceRange.layerCount = 1;

	vk::MemoryRequirements memReqs;

	depthStencil.image = device.createImage(image, nullptr);
	memReqs = device.getImageMemoryRequirements(depthStencil.image);
	mem_alloc.allocationSize = memReqs.size;
	mem_alloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
	depthStencil.mem = device.allocateMemory(mem_alloc, nullptr);
	
	device.bindImageMemory(depthStencil.image, depthStencil.mem, 0);
	vkTools::setImageLayout(
		setupCmdBuffer,
		depthStencil.image,
		vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthStencilAttachmentOptimal);

	depthStencilView.image = depthStencil.image;
	depthStencil.view = device.createImageView(depthStencilView, nullptr);
}

void VulkanExampleBase::setupFrameBuffer()
{
	vk::ImageView attachments[2];

	// Depth/Stencil attachment is the same for all frame buffers
	attachments[1] = depthStencil.view;

	vk::FramebufferCreateInfo frameBufferCreateInfo = {};
	frameBufferCreateInfo.renderPass = renderPass;
	frameBufferCreateInfo.attachmentCount = 2;
	frameBufferCreateInfo.pAttachments = attachments;
	frameBufferCreateInfo.width = width;
	frameBufferCreateInfo.height = height;
	frameBufferCreateInfo.layers = 1;

	// Create frame buffers for every swap chain image
	frameBuffers.resize(swapChain.imageCount);
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		attachments[0] = swapChain.buffers[i].view;
		frameBuffers[i] = device.createFramebuffer(frameBufferCreateInfo, nullptr);
	}
}

void VulkanExampleBase::setupRenderPass()
{
	vk::AttachmentDescription attachments[2] = {};

	// Color attachment
	attachments[0].format = colorformat;
	attachments[0].samples = vk::SampleCountFlagBits::e1;
	attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
	attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
	attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	attachments[0].initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachments[0].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

	// Depth attachment
	attachments[1].format = depthFormat;
	attachments[1].samples = vk::SampleCountFlagBits::e1;
	attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
	attachments[1].storeOp = vk::AttachmentStoreOp::eStore;
	attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	attachments[1].initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
	attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	vk::AttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::AttachmentReference depthReference = {};
	depthReference.attachment = 1;
	depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	vk::SubpassDescription subpass = {};
	subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.inputAttachmentCount = 0;
	subpass.pInputAttachments = NULL;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorReference;
	subpass.pResolveAttachments = NULL;
	subpass.pDepthStencilAttachment = &depthReference;
	subpass.preserveAttachmentCount = 0;
	subpass.pPreserveAttachments = NULL;

	vk::RenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = vk::StructureType::eRenderPassCreateInfo;
	renderPassInfo.pNext = NULL;
	renderPassInfo.attachmentCount = 2;
	renderPassInfo.pAttachments = attachments;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 0;
	renderPassInfo.pDependencies = NULL;

	renderPass = device.createRenderPass(renderPassInfo, nullptr);
}

void VulkanExampleBase::windowResize()
{
	if (!prepared)
	{
		return;
	}
	prepared = false;

	// Recreate swap chain
	width = destWidth;
	height = destHeight;
	createSetupCommandBuffer();
	setupSwapChain();

	// Recreate the frame buffers

	device.destroyImageView(depthStencil.view, nullptr);
	device.destroyImage(depthStencil.image, nullptr);
	device.freeMemory(depthStencil.mem, nullptr);
	setupDepthStencil();
	
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		device.destroyFramebuffer(frameBuffers[i], nullptr);
	}
	setupFrameBuffer();

	flushSetupCommandBuffer();

	// Command buffers need to be recreated as they may store
	// references to the recreated frame buffer
	destroyCommandBuffers();
	createCommandBuffers();
	buildCommandBuffers();

	queue.waitIdle();
	vkDeviceWaitIdle(device);

	if (enableTextOverlay)
	{
		textOverlay->reallocateCommandBuffers();
		updateTextOverlay();
	}

	// Notify derived class
	windowResized();
	viewChanged();

	prepared = true;
}

void VulkanExampleBase::windowResized()
{
	// Can be overriden in derived class
}

void VulkanExampleBase::initSwapchain()
{
#if defined(_WIN32)
	swapChain.initSurface(windowInstance, window);
#elif defined(__ANDROID__)	
	swapChain.initSurface(androidApp->window);
#elif defined(__linux__)
	swapChain.initSurface(connection, window);
#endif
}

void VulkanExampleBase::setupSwapChain()
{
	swapChain.create(setupCmdBuffer, &width, &height);
}
