//const FULLSCREEN = false;

const CANVAS_WIDTH = 1920;
const CANVAS_HEIGHT = 1080;

let audioContext;
let audioBufferSourceNode;

let device;
let uniformBuffer;
let rulesBuffer;
let bindGroup = [];
let computePipeline;
let renderPipeline;
let renderPassDescriptor;

let canvas;
let context;

let startTime;
let lastSimulationUpdateTime = 0;
let simulationIteration = 0;
let activeScene = -1;

const RULES = new Uint32Array([
   5,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0, // clouds-5
   5,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 4/4-5
   5,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // amoeba-5
  10,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // pyro-10
   5,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // framework-5
  10,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // spiky-10
  10,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0, // ripple-10
]);

// Time, rule, delta, radius
const SCENES = [
  0, 2, 0.1, 25, // amoeba
  19, 3, 0.25, 320, // pyro
  30, 2, 0.3, 220, // amoeba
  40, 0, 0.5, 180, // clouds
  55, 6, 0.375, 160, // ripple
  75, 3, 0.437, 190, // pyro (trim down)
  77, 4, 0.1, 180, // framework
  95, 5, 0.15, 160, // spiky
  110, 1, 0.2, 150, // 445
  150, 1, 0.2, 190
];

// Will be replaced by compression script
const AUDIO_SHADER = `
REPLACE_ME_AUDIO
`;

const VISUAL_SHADER = `
REPLACE_ME_VISUAL
`;

async function createComputePipeline(shaderModule, pipelineLayout)
{
  return device.createComputePipelineAsync({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: "C"
    }
  });
}

async function createRenderPipeline(shaderModule, pipelineLayout)
{
  return device.createRenderPipelineAsync({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "V"
    },
    fragment: {
      module: shaderModule,
      entryPoint: "F",
      targets: [{format: "bgra8unorm"}]
    },
    primitive: {topology: "triangle-strip"}
  });
}

function encodeComputePassAndSubmit(commandEncoder, pipeline, bindGroup)
{
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(64, 64, 64);
  passEncoder.end();
}

function encodeRenderPassAndSubmit(commandEncoder, pipeline, bindGroup, passDescriptor)
{
  const passEncoder = commandEncoder.beginRenderPass(passDescriptor);
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.draw(4);
  passEncoder.end();
}

async function createAudioResources()
{
  audioContext = new AudioContext();
  let webAudioBuffer = audioContext.createBuffer(2, 256 ** 3, audioContext.sampleRate);

  let audioBuffer = device.createBuffer({
    size: (256 ** 3) * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});

  // Will be reused by visual shader
  uniformBuffer = device.createBuffer({
    size: 4 * 4, // We actually only need 1 float for audio
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  let audioBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}}, // Was uniform
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
    ]});

  let audioBindGroup = device.createBindGroup({
    layout: audioBindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: uniformBuffer}},
      {binding: 1, resource: {buffer: audioBuffer}}
    ]});

  let audioReadBuffer = device.createBuffer({
    size: (256 ** 3) * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});

  let audioPipelineLayout = device.createPipelineLayout({bindGroupLayouts: [audioBindGroupLayout]});

  device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([audioContext.sampleRate]));

  let pipeline = await createComputePipeline(device.createShaderModule({code: AUDIO_SHADER}), audioPipelineLayout);

  let commandEncoder = device.createCommandEncoder();

  encodeComputePassAndSubmit(commandEncoder, pipeline, audioBindGroup);

  commandEncoder.copyBufferToBuffer(audioBuffer, 0, audioReadBuffer, 0, (256 ** 3) * 8);

  device.queue.submit([commandEncoder.finish()]);

  await audioReadBuffer.mapAsync(GPUMapMode.READ);
  let audioData = new Float32Array(audioReadBuffer.getMappedRange());

  let channel0 = webAudioBuffer.getChannelData(0);
  let channel1 = webAudioBuffer.getChannelData(1);

  for(let i=0; i<256 ** 3; i++) {
    channel0[i] = audioData[(i << 1) + 0];
    channel1[i] = audioData[(i << 1) + 1];
  }

  audioReadBuffer.unmap();

  audioBufferSourceNode = audioContext.createBufferSource();
  audioBufferSourceNode.buffer = webAudioBuffer;
  audioBufferSourceNode.connect(audioContext.destination); 
}

async function createRenderResources()
{
  let bindGroupLayout = device.createBindGroupLayout({
    entries: [ 
      {binding: 0, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}}, // Was uniform
      {binding: 1, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT, buffer: {type: "read-only-storage"}},
    ]
  });

  let gridBuffer = [];
  for(let i=0; i<2; i++)
    gridBuffer[i] = device.createBuffer({
      size: (256 ** 3) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  rulesBuffer = device.createBuffer({
    size: 385 * 4, //RULES.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});

  for(let i=0; i<2; i++) {
    bindGroup[i] = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {binding: 0, resource: {buffer: uniformBuffer}},
        {binding: 1, resource: {buffer: gridBuffer[i]}},
        {binding: 2, resource: {buffer: gridBuffer[1 - i]}},
        {binding: 3, resource: {buffer: rulesBuffer}},
      ]
    });
  }

  let pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]});

  renderPassDescriptor = {
    colorAttachments: [{
      0, // view
      //clearValue: {r: 1.0, g: 0.0, b: 0.0, a: 1.0},
      loadOp: "clear",
      storeOp: "store"
    }]
  };

  let shaderModule = device.createShaderModule({code: VISUAL_SHADER});
  computePipeline = await createComputePipeline(shaderModule, pipelineLayout);
  renderPipeline = await createRenderPipeline(shaderModule, pipelineLayout);
  
  // Set grid
  const area = 24;
  const pos = 128 - area / 2;
  let grid = new Uint32Array(area);
  for(let k=0; k<area; k++) {
    for(let j=0; j<area; j++) { 
      for(let i=0; i<area; i++)
        grid[i] = Math.min(1, RULES[(area * (k ^ j) + (i ^ k)) % 385]);
      let ofs = (256 ** 2) * (pos + k) + 256 * (pos + j) + pos;
      device.queue.writeBuffer(gridBuffer[0], ofs * 4, grid);
      device.queue.writeBuffer(gridBuffer[1], ofs * 4, grid);
    }
  }
}

//let last; // frame time

function render(time)
{  
  /*if(last !== undefined) {
    let frameTime = (performance.now() - last);
    document.title = `${(frameTime).toFixed(2)} / ${(1000.0 / frameTime).toFixed(2)}`;
  }
  last = performance.now();*/

  // Initialize time and start audio
  if(startTime === undefined) {
    audioBufferSourceNode.start(0, 0);
    startTime = audioContext.currentTime;
  }

  // Current time
  time = audioContext.currentTime - startTime;
  if(time >= 150) //SCENES.at(-1).t
    return;

  // Scene update
  if(time >= SCENES[4 * (activeScene + 1)])
    device.queue.writeBuffer(rulesBuffer, 0, RULES, SCENES[ 4 * (++activeScene) + 1 ] * 55, 55);

  // Current scene time
  let t = (time - SCENES[4 * activeScene]) / (SCENES[4 * (activeScene + 1)] - SCENES[4 * activeScene]);

  const commandEncoder = device.createCommandEncoder();
  
  // Simulation
  if(time - lastSimulationUpdateTime > SCENES[4 * activeScene + 2]) {
    encodeComputePassAndSubmit(commandEncoder, computePipeline, bindGroup[simulationIteration % 2]); 
    simulationIteration++;
    lastSimulationUpdateTime = audioContext.currentTime - startTime;
  }

  // Camera
  device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
    SCENES[4 * activeScene + 3] + (SCENES[4 * (activeScene + 1) + 3] - SCENES[4 * activeScene + 3]) * t, // radius
    ((activeScene % 2) ? 1 : -1) * t * 2 * Math.PI, // phi
    (0.6 + 0.3 * Math.sin(time * 0.4)) * Math.sin(time * 0.1), // theta
    time]));

  // Render
  renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
  encodeRenderPassAndSubmit(commandEncoder, renderPipeline, bindGroup[simulationIteration % 2], renderPassDescriptor);
  
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

function startRender()
{
  document.querySelector("button").removeEventListener("click", startRender);

  //if(FULLSCREEN)
    canvas.requestFullscreen();
  /*else {
    canvas.style.width = CANVAS_WIDTH;
    canvas.style.height = CANVAS_HEIGHT;
    canvas.style.position = "absolute";
    canvas.style.left = 0;
    canvas.style.top = 0;
  }*/

  requestAnimationFrame(render);
}

async function main()
{
  if(!navigator.gpu)
    alert("No WebGPU");

  const gpuAdapter = await navigator.gpu.requestAdapter();
  device = await gpuAdapter.requestDevice();

  await createAudioResources();
  await createRenderResources();

  document.body.innerHTML = "<button>CLICK<canvas style='width:0;cursor:none'>";
  canvas = document.querySelector("canvas");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;

  context = canvas.getContext("webgpu");
  context.configure({device, format: "bgra8unorm", alphaMode: "opaque"});

  document.querySelector("button").addEventListener("click", startRender);
}

main();
