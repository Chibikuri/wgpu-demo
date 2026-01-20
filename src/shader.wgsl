// Neural Network Visualization Shader
// An AI-themed demo showing animated neural network with flowing activations

struct Uniforms {
    time: f32,
    _pad1: f32,
    resolution: vec2<f32>,
    _pad2: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = positions[vertex_index] * 0.5 + 0.5;
    return out;
}

// AI Color Palette
fn ai_blue() -> vec3<f32> { return vec3<f32>(0.2, 0.6, 1.0); }
fn ai_purple() -> vec3<f32> { return vec3<f32>(0.6, 0.3, 0.9); }
fn ai_cyan() -> vec3<f32> { return vec3<f32>(0.3, 0.9, 0.9); }
fn ai_pink() -> vec3<f32> { return vec3<f32>(0.9, 0.3, 0.6); }

// Smooth minimum for blending shapes
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Circle SDF
fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// Line segment SDF
fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// Hash function for randomness
fn hash(p: vec2<f32>) -> f32 {
    let p2 = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(dot(p2, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Neural network node positions for each layer
fn get_node_pos(layer: i32, node: i32, total_nodes: i32) -> vec2<f32> {
    let layer_x = f32(layer) * 0.22 - 0.33;
    let node_spacing = 1.4 / f32(total_nodes + 1);
    let node_y = f32(node + 1) * node_spacing - 0.7;
    return vec2<f32>(layer_x, node_y);
}

// Draw a neuron with activation glow
fn draw_neuron(uv: vec2<f32>, center: vec2<f32>, activation: f32, time: f32) -> vec3<f32> {
    let d = length(uv - center);
    let radius = 0.025;

    // Core of the neuron
    let core = smoothstep(radius, radius - 0.008, d);

    // Activation glow
    let glow_size = radius + 0.04 * activation;
    let glow = smoothstep(glow_size, 0.0, d) * activation * 0.6;

    // Pulse effect
    let pulse = sin(time * 3.0 + activation * 6.28) * 0.5 + 0.5;
    let pulse_glow = smoothstep(glow_size + 0.02, 0.0, d) * pulse * activation * 0.3;

    // Color based on activation
    let base_color = mix(ai_blue(), ai_cyan(), activation);
    let glow_color = mix(ai_purple(), ai_pink(), pulse);

    return base_color * core + glow_color * (glow + pulse_glow);
}

// Draw connection between neurons with data flow animation
fn draw_connection(uv: vec2<f32>, start: vec2<f32>, end: vec2<f32>, weight: f32, time: f32, phase: f32) -> vec3<f32> {
    let d = sd_segment(uv, start, end);

    // Calculate position along the connection for flow animation
    let dir = normalize(end - start);
    let len = length(end - start);
    let proj = dot(uv - start, dir);
    let t = proj / len;

    // Data pulse traveling along the connection
    let flow_speed = 1.5;
    let pulse_pos = fract(time * flow_speed + phase);
    let pulse_width = 0.15;
    let pulse = smoothstep(pulse_width, 0.0, abs(t - pulse_pos)) * weight;

    // Base connection line
    let line_width = 0.003 + weight * 0.004;
    let line = smoothstep(line_width + 0.002, line_width, d);

    // Flow glow
    let flow_glow = smoothstep(0.025, 0.0, d) * pulse;

    let base_color = ai_blue() * 0.3 * weight;
    let flow_color = mix(ai_cyan(), ai_pink(), sin(time + phase * 6.28) * 0.5 + 0.5);

    return base_color * line + flow_color * flow_glow;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Aspect ratio correction
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    var uv = in.uv * 2.0 - 1.0;
    uv.x *= aspect;

    let time = uniforms.time;
    var color = vec3<f32>(0.02, 0.02, 0.05); // Dark background

    // Neural network structure: 4 layers with varying nodes
    let layers = array<i32, 4>(3, 5, 5, 2);

    // Draw connections first (behind nodes)
    for (var l = 0; l < 3; l++) {
        let current_layer_size = layers[l];
        let next_layer_size = layers[l + 1];

        for (var i = 0; i < current_layer_size; i++) {
            for (var j = 0; j < next_layer_size; j++) {
                let node_start = get_node_pos(l, i, current_layer_size);
                let node_end = get_node_pos(l + 1, j, next_layer_size);

                // Weight based on pseudo-random value
                let weight = hash(vec2<f32>(f32(l * 100 + i), f32(j))) * 0.5 + 0.5;
                let phase = hash(vec2<f32>(f32(i * 7), f32(j * 13 + l)));

                color += draw_connection(uv, node_start, node_end, weight, time, phase);
            }
        }
    }

    // Draw neurons
    for (var l = 0; l < 4; l++) {
        let layer_size = layers[l];
        for (var n = 0; n < layer_size; n++) {
            let pos = get_node_pos(l, n, layer_size);

            // Activation varies with time and position
            let base_activation = hash(vec2<f32>(f32(l), f32(n)));
            let activation = (sin(time * 2.0 + base_activation * 6.28 + f32(l) * 0.5) * 0.5 + 0.5)
                           * (0.5 + base_activation * 0.5);

            color += draw_neuron(uv, pos, activation, time);
        }
    }

    // Add subtle grid pattern (circuit board effect)
    let grid_size = 0.05;
    let grid = smoothstep(0.002, 0.0, abs(fract(uv.x / grid_size) - 0.5) - 0.48)
             + smoothstep(0.002, 0.0, abs(fract(uv.y / grid_size) - 0.5) - 0.48);
    color += ai_blue() * grid * 0.03;

    // Vignette effect
    let vignette = 1.0 - length(in.uv - 0.5) * 0.8;
    color *= vignette;

    // Add floating particles (data points)
    for (var i = 0; i < 8; i++) {
        let seed = vec2<f32>(f32(i) * 1.7, f32(i) * 2.3);
        let particle_x = sin(time * 0.3 + seed.x * 3.0) * 0.6;
        let particle_y = fract(time * 0.1 + seed.y * 0.5) * 2.0 - 1.0;
        let particle_pos = vec2<f32>(particle_x, particle_y);

        let d = length(uv - particle_pos);
        let particle = smoothstep(0.02, 0.0, d) * 0.3;
        color += mix(ai_cyan(), ai_purple(), hash(seed)) * particle;
    }

    // Gamma correction
    color = pow(color, vec3<f32>(0.8));

    return vec4<f32>(color, 1.0);
}
