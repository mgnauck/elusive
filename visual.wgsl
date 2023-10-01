struct Hit
{
  index: i32,
  norm: vec3f,
  state: u32,
  dist: f32,
}

const GRID_MUL = vec3i(1, 256, 256 * 256);
const GRID_MUL_F = vec3f(GRID_MUL);

@group(0) @binding(0) var<storage> uniforms: array<f32>;
@group(0) @binding(1) var<storage> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputGrid: array<u32>;
@group(0) @binding(3) var<storage> rules: array<u32>;

fn getCell(x: i32, y: i32, z: i32) -> u32
{
  // Consider only states 0 and 1. Cells in refactory period do NOT count as active neighbours, i.e. are counted as 0.
  return u32(1 - min(abs(1 - i32(grid[GRID_MUL.z * z + GRID_MUL.y * y + x])), 1));
}

fn getMooreNeighbourCountWrap(pos: vec3i) -> u32
{
  let dec = vec3i((pos.x - 1) % GRID_MUL.y, (pos.y - 1) % GRID_MUL.y, (pos.z - 1) % GRID_MUL.y);
  let inc = vec3i((pos.x + 1) % GRID_MUL.y, (pos.y + 1) % GRID_MUL.y, (pos.z + 1) % GRID_MUL.y);

  return  getCell(pos.x, inc.y, pos.z) +
          getCell(inc.x, inc.y, pos.z) +
          getCell(dec.x, inc.y, pos.z) +
          getCell(pos.x, inc.y, inc.z) +
          getCell(pos.x, inc.y, dec.z) +
          getCell(inc.x, inc.y, inc.z) +
          getCell(inc.x, inc.y, dec.z) +
          getCell(dec.x, inc.y, inc.z) +
          getCell(dec.x, inc.y, dec.z) +
          getCell(inc.x, pos.y, pos.z) +
          getCell(dec.x, pos.y, pos.z) +
          getCell(pos.x, pos.y, inc.z) +
          getCell(pos.x, pos.y, dec.z) +
          getCell(inc.x, pos.y, inc.z) +
          getCell(inc.x, pos.y, dec.z) +
          getCell(dec.x, pos.y, inc.z) +
          getCell(dec.x, pos.y, dec.z) +
          getCell(pos.x, dec.y, pos.z) +
          getCell(inc.x, dec.y, pos.z) +
          getCell(dec.x, dec.y, pos.z) +
          getCell(pos.x, dec.y, inc.z) +
          getCell(pos.x, dec.y, dec.z) +
          getCell(inc.x, dec.y, inc.z) +
          getCell(inc.x, dec.y, dec.z) +
          getCell(dec.x, dec.y, inc.z) +
          getCell(dec.x, dec.y, dec.z);
}

@compute @workgroup_size(4,4,4)
fn C(@builtin(global_invocation_id) globalId: vec3u)
{
  let pos = vec3i(globalId);
  let index = dot(pos, GRID_MUL);
  let value = grid[index];

  if value == 0 {
    outputGrid[index] = rules[28 + getMooreNeighbourCountWrap(pos)];
  } else if value == 1 {
    // Dying state 1 goes to 2 (or dies directly by being moduloed to 0, in case there are only 2 states)
    outputGrid[index] = (2 - rules[1 + getMooreNeighbourCountWrap(pos)]) % rules[0];
  } else {
    // Refactory period
    outputGrid[index] = min(value + 1, rules[0]) % rules[0]; 
  }
}

fn minComp(v: vec3f) -> f32
{
  return min(v.x, min(v.y, v.z));
}

fn maxComp(v: vec3f) -> f32
{
  return max(v.x, max(v.y, v.z));
}

fn intersectAabb(minExt: vec3f, maxExt: vec3f, ori: vec3f, invDir: vec3f, tmin: ptr<function, f32>, tmax: ptr<function, f32>) -> bool
{
  let t0 = (minExt - ori) * invDir;
  let t1 = (maxExt - ori) * invDir;
  
  *tmin = maxComp(min(t0, t1));
  *tmax = minComp(max(t0, t1));

  return *tmin <= *tmax && *tmax > 0;
}

fn traverseGrid(ori: vec3f, invDir: vec3f, tmax: f32, hit: ptr<function, Hit>) -> bool
{
  var stepDir = sign(invDir);
  var t = (vec3f(0.5) + 0.5 * stepDir - fract(ori)) * invDir;
  var mask = vec3f(0);

  (*hit).dist = 0;
  (*hit).index = i32(dot(GRID_MUL_F, floor(vec3f(GRID_MUL_F.y * 0.5) + ori)));
 
  var iter = 0;
  loop {
    if iter > 512 { // Nvidia fix :(
      return false;
    }

    (*hit).state = grid[(*hit).index];
    if (*hit).state > 0 {
      (*hit).norm = mask * -stepDir;
      return true;
    }

    (*hit).dist = minComp(t);
    if (*hit).dist >= tmax {
      return false;
    }
 
    mask.x = f32(t.x <= t.y && t.x <= t.z);
    mask.y = f32(t.y <= t.x && t.y <= t.z);
    mask.z = f32(t.z <= t.x && t.z <= t.y);

    t += mask * stepDir * invDir;
    (*hit).index += i32(dot(GRID_MUL_F, mask * stepDir));
    
    iter++;
  }
}

fn calcOcclusion(pos: vec3f, index: i32, norm: vec3i) -> f32
{
  let above = index + dot(GRID_MUL, norm);
  let dir = abs(norm);
  let hori = dot(GRID_MUL, dir.yzx);
  let vert = dot(GRID_MUL, dir.zxy);

  let edgeCellStates = vec4f(
    f32(min(1, grid[above + hori])),
    f32(min(1, grid[above - hori])),
    f32(min(1, grid[above + vert])),
    f32(min(1, grid[above - vert])));

  let cornerCellStates = vec4f(
    f32(min(1, grid[above + hori + vert])),
    f32(min(1, grid[above - hori + vert])),
    f32(min(1, grid[above + hori - vert])),
    f32(min(1, grid[above - hori - vert])));

  let uvLocal = fract(pos);
  let uv = vec2f(dot(uvLocal, vec3f(dir.yzx)), dot(uvLocal, vec3f(dir.zxy)));
  let uvInv = vec2f(1) - uv;

  let edgeOcc = edgeCellStates * vec4f(uv.x, uvInv.x, uv.y, uvInv.y);
  let cornerOcc = cornerCellStates * vec4f(uv.x * uv.y, uvInv.x * uv.y, uv.x * uvInv.y, uvInv.x * uvInv.y) * (vec4f(1) - edgeCellStates.xzwy) * (vec4f(1) - edgeCellStates.zyxw);

  return 1 - (edgeOcc.x + edgeOcc.y + edgeOcc.z + edgeOcc.w + cornerOcc.x + cornerOcc.y + cornerOcc.z + cornerOcc.w) * 0.5;
}

fn shade(pos: vec3f, tmax: f32, hit: ptr<function, Hit>) -> vec3f
{
  let cnt = f32(rules[0]);
  let val = cnt / min(f32((*hit).state), cnt);
  let sky = 0.4 + (*hit).norm.y * 0.6;
  let col = vec3f(0.005) + (1 - 0.15 * (cnt - 5)) * (vec3f(0.5) + pos / GRID_MUL_F.y) * sky * sky * val * val * 0.3 * exp(-3.5 * (*hit).dist / tmax);
  let occ = calcOcclusion(pos, (*hit).index, vec3i((*hit).norm));

  return col * occ * occ * occ;
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn filmicToneACES(x: vec3f) -> vec3f
{
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return saturate(x * (a * x + vec3(b)) / (x * (c * x + vec3(d)) + vec3(e)));
}

@vertex
fn V(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f
{
  let pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1), vec2f(1), vec2f(1, -1));
  return vec4f(pos[vertexIndex], 0, 1);
}

@fragment
fn F(@builtin(position) pos: vec4f) -> @location(0) vec4f
{
  let dirEyeSpace = normalize(vec3f((pos.xy - vec2f(1920, 1080) * 0.5) / 1080, 1 /* FOV 50 */));
  //let dirEyeSpace = normalize(vec3f((pos.xy - vec2f(1024, 578) * 0.5) / 578, 1 /* FOV 50 */));

  let ori = vec3f(uniforms[0] * cos(uniforms[2]) * cos(uniforms[1]), uniforms[0] * sin(uniforms[2]), uniforms[0] * cos(uniforms[2]) * sin(uniforms[1]));

  let fwd = normalize(-ori);
  let ri = normalize(cross(fwd, vec3f(0, 1, 0)));

  var dir = ri * dirEyeSpace.x - cross(ri, fwd) * dirEyeSpace.y + fwd * dirEyeSpace.z;

  let halfGrid = vec3f(GRID_MUL_F.y * 0.5);
  let invDir = 1 / dir;

  var tmin: f32;
  var tmax: f32;
  var hit: Hit;
  var col = vec3f(0);

  if intersectAabb(-halfGrid, halfGrid, ori, invDir, &tmin, &tmax) {
    tmin = max(tmin + 0.001, 0); // Epsilon
    tmax = tmax - 0.001 - tmin;
    if traverseGrid(ori + tmin * dir, invDir, tmax, &hit) {
      col = shade(ori + (tmin + hit.dist) * dir, tmax, &hit);
    }
  }

  return vec4f(pow(filmicToneACES(mix(col, vec3f(0), 1 - smoothstep(0, 15, uniforms[3]) + smoothstep(135, 150, uniforms[3]))), vec3f(0.4545)), 1);
}
