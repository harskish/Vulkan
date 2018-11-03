#version 460
#extension GL_NVX_raytracing : require

// Aligned as vec4
struct Vertex {
	vec3 pos;
	vec3 normal;
	vec3 color;
};

//layout(binding = 5) uniform accelerationStructureNVX bvh;

layout(location = 0) rayPayloadInNVX Payload {
	vec3 color;
} payload;

layout(location = 1) rayPayloadNVX ShadowPayload {
	bool blocked;
} shadowPayload;

layout(location = 1) hitAttributeNVX vec3 attribs;

layout(std430, binding = 3) readonly buffer Indices {
    uint indices[];
};

layout(std430, binding = 4) readonly buffer Vertices {
    Vertex vertices[];
};

const vec3 lightPos = vec3(0.0, 0.0, 0.0);
const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const float lightIntensity = 1.0;

void main() {
	vec3 bar = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	uint i0 = uint(indices[gl_PrimitiveID * 3 + 0]);
	uint i1 = uint(indices[gl_PrimitiveID * 3 + 1]);
	uint i2 = uint(indices[gl_PrimitiveID * 3 + 2]);

	vec3 C0 = vertices[i0].color;
	vec3 C1 = vertices[i1].color;
	vec3 C2 = vertices[i2].color;
	vec3 C = bar.x * C0 + bar.y * C1 + bar.z * C2;

	vec3 N0 = vertices[i0].normal;
	vec3 N1 = vertices[i1].normal;
	vec3 N2 = vertices[i2].normal;
	vec3 N = bar.x * N0 + bar.y * N1 + bar.z * N2;

	vec3 dirIn = gl_WorldRayDirectionNVX;
	dirIn.y *= -1.0;

	vec3 posWorld = gl_WorldRayOriginNVX + gl_WorldRayDirectionNVX * gl_HitTNVX;
	vec3 L = lightPos - posWorld;
	float lightDist = length(L);

	shadowPayload.blocked = false;
	//traceNVX(bvh, gl_RayFlagsOpaqueNVX, 0xff, 0, 0, 0, posWorld, 1e-3f, normalize(L), lightDist, 1);

	if (!shadowPayload.blocked)
		payload.color = abs(dot(L, N)) * C * lightColor * lightIntensity / (lightDist * lightDist);
	else
		payload.color = abs(dot(L, N)) * C * 0.001;
}
