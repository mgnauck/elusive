const PI = 3.1415;
const TAU = 2 * PI;
const TIME_PER_BEAT = 0.12; // 60.0 / 125.0 / 4.0;
const TIME_PER_PATTERN = 1.92; // 60.0 / 125.0 * 4.0;
const PATTERN_COUNT = 120;
const KICK = 0;
const HIHAT = 1;
const BASS = 2;
const DURCH = 3;

// Bad rand, ripped from shader toy or somewhere
fn noise(phase: f32) -> vec4f
{
  let uv = phase / vec2f(0.512, 0.487);
  return vec4f(fract(sin(dot(uv, vec2f(12.9898, 78.233))) * 43758.5453));
}

fn hihat(time: f32, freq: f32) -> f32
{
  return atan2(noise(time * freq).x, 0.15) * exp(-60 * time); 
}

fn bass3(time: f32, freq: f32) -> f32
{
  if(time < 0)
  {
    return 0;
  }

  return sin(time * TAU * freq) * smoothstep(0, 1, time * 16) * exp(-1 * time);
}

// Inspired by https://www.shadertoy.com/view/7lpatternIndexczz
fn kick(time: f32, freq: f32) -> f32
{
  if time < 0
  {
    return 0;
  }

  let phase = freq * time - 8 * exp(-20 * time) - 3 * exp(-800 * time);
  return atan2(sin(TAU * phase), 0.4) * exp(-5 * time);
}

fn sample1(gTime: f32, time: f32, freq: f32) -> f32
{
  let lfo = sin(gTime * TAU * 0.001) * 0.5;
  let lfo2 = sin(gTime * TAU * lfo) * 0.5;

  var out = 0.0;
  for(var v=1.0; v<=11; v+=1)
  {
    let lfo3 = 0.1 * sin(gTime * v * TAU * 0.05);
    let detune = lfo3 * v;
    let f0 = 0.5 * freq * v;
    out += 0.1 * sin(TAU * time * (f0 + detune + lfo - 55));
  }

  return atan2(out, 1 - lfo2 * 0.2) * smoothstep(0, 1, time * 8) * exp(-2 * time);
}

fn sample1lpf(gTime: f32, time: f32, freq: f32) -> f32
{
  let c = 0.8 - sin(gTime * TAU * 0.001) * 0.3;
  let r = 0.8 - cos(gTime * TAU * 0.1) * 0.4;
  let dt = 1 / f32(params[0]);

  var v0 = 0.0;
  var v1 = 0.0;
  for(var j=0.0; j<96; j += 1)
  {
    let input = sample1(gTime, time - j * dt, freq);
    v0 = (1 - r * c) * v0 - c * v1 + c * input;
    v1 = (1 - r * c) * v1 + c * v0;
  }
  
  return v1;
}

fn addSample(idx: u32, gTime: f32, time: f32, pat: f32, dur: f32, freq: f32, amp: f32) -> f32
{
  let sampleTime = time - pat * TIME_PER_BEAT;

  if sampleTime < 0 || sampleTime > dur
  {
    return 0;
  }

  // If the duration causes the sound to shutdown we want at least a quick
  // ramp down to zero to not cause a click this seems to work but better double check again
  let env = amp * smoothstep(0, 0.05, dur - sampleTime);

  if idx == KICK
  {
    return kick(sampleTime, freq) * env;
  }
  else if idx == HIHAT
  {
    return hihat(sampleTime, freq) * env;
  }
  else if idx == BASS
  {
    return bass3(sampleTime, freq) * env;
  }
  else if idx == DURCH
  {
    return sample1lpf(gTime, sampleTime, freq) * env;
  }
  
  return 0;
}

fn isPattern(time: f32, start: u32, end: u32) -> bool
{
  let patternIndex = u32(time / TIME_PER_PATTERN) % PATTERN_COUNT; 

  return patternIndex >= start && patternIndex < end;
}

@group(0) @binding(0) var<storage> params: array<u32>; // params[0] = sample rate
@group(0) @binding(1) var<storage, read_write> buffer: array<vec2f>;

@compute @workgroup_size(4, 4, 4)
fn C(@builtin(global_invocation_id) globalId: vec3u)
{
  // Calculate current sample from given buffer id
  let sample = dot(globalId, vec3u(1, 256, 256 * 256));
  let time = f32(sample) / f32(params[0]);
  let patternTime = time % TIME_PER_PATTERN;

  // Samples are calculated in mono and then written to left/right
  var output = vec2f(0);

  // 60/125*120 = 57,6 = 58 patterns
  if isPattern(time, 4, PATTERN_COUNT)
  {
    output += addSample(DURCH, time, patternTime, 0, 0.5, 55, 0.8 );
  }

  // always
  output += addSample(DURCH, time, patternTime,  2, 1.0, 110, 0.9 );    

  // bass
  if isPattern(time, 10, PATTERN_COUNT)
  {
    output += addSample(BASS, time, patternTime,   2, 0.25, 110, 0.4 );
    output += addSample(BASS, time, patternTime,   6, 0.25, 110, 0.3 );
    output += addSample(BASS, time, patternTime,  10, 0.25, 110, 0.4 );
    output += addSample(BASS, time, patternTime,  14, 0.125, 110, 0.2 );
  }

  // hihat + kick
  if isPattern(time, 10, 11)
  {
    output += addSample(KICK, time, patternTime,  14, 1, 55, 0.5 );
  }

  if isPattern(time, 10, PATTERN_COUNT)
  {
    output += addSample(KICK, time, patternTime,  0, 1, 55, 0.4 );
    output += addSample(KICK, time, patternTime,  4, 1, 55, 0.5 );
    output += addSample(KICK, time, patternTime,  8, 1, 55, 0.4 );
    output += addSample(KICK, time, patternTime, 12, 1, 55, 0.5 );
    output += addSample(HIHAT, time, patternTime,  0, 0.1, 55, 0.3);
    output += addSample(HIHAT, time, patternTime,  4, 0.1, 55, 0.15);
    output += addSample(HIHAT, time, patternTime,  8, 0.1, 55, 0.20);
    output += addSample(HIHAT, time, patternTime, 12, 0.1, 55, 0.15);
    output += addSample(HIHAT, time, patternTime, 15, 0.2, 55, 0.13);
  }

  // Party special
  if time >= 75 && time < 77
  {
    output += 0.1 * noise(time).xy;
  }

  // Global fade in-/out
  output *= mix(0, smoothstep(0, 3, time), smoothstep(150, 138, time));

  // Write 2 floats between -1 and 1 to output buffer (stereo)
  buffer[sample] = clamp(output, -vec2f(1), vec2f(1));
}
