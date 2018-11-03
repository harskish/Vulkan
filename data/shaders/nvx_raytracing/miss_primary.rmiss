#version 460
#extension GL_NVX_raytracing : require

layout(location = 0) rayPayloadInNVX Payload {
	vec3 color;
	bool isShadowRay;
	bool shadowRayBlocked;
} payload;

/* Primary ray miss shader */

void main() {
    payload.color = vec3(0.2, 0.2, 0.2);
}
