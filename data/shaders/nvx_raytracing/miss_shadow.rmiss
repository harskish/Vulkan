#version 460
#extension GL_NVX_raytracing : require

layout(location = 1) rayPayloadInNVX ShadowPayload {
	bool blocked;
} shadowPayload;

/* Shadow ray that misses all geometry */

void main() {
    shadowPayload.blocked = false;
}
