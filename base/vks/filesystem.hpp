#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#endif

namespace vks { namespace file {
#if defined(__ANDROID__)
    void setAssetManager(AAssetManager* assetManager);
#endif

    void withBinaryFileContexts(const std::string& filename, std::function<void(const char* filename, size_t size, const void* data)> handler);

    void withBinaryFileContexts(const std::string& filename, std::function<void(size_t size, const void* data)> handler);

    //std::vector<uint8_t> readBinaryFile(const std::string& filename);

    std::string readTextFile(const std::string& fileName);

} }
