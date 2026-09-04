import { describe, expect, it } from "vitest";
import { BWLabeler } from "./bwlabels.js";

// Build a flat Uint8Array volume from a 3D array indexed as [slice][col][row],
// matching BWLabeler.idx: addr = C*dim0*dim1 + B*dim0 + A (A=row, B=col, C=slice).
function volume(slices: number[][][]) {
  const dimZ = slices.length;
  const dimY = slices[0].length;
  const dimX = slices[0][0].length;
  const img = new Uint8Array(dimX * dimY * dimZ);
  for (let c = 0; c < dimZ; c++) {
    for (let b = 0; b < dimY; b++) {
      for (let a = 0; a < dimX; a++) {
        img[c * dimX * dimY + b * dimX + a] = slices[c][b][a];
      }
    }
  }
  return { img, dim: [dimX, dimY, dimZ] };
}

describe("BWLabeler.bwlabel", () => {
  it("labels two disconnected blobs in a single slice as two clusters", () => {
    // 4x4x1 with two 1-voxel blobs separated by background.
    const { img, dim } = volume([
      [
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
      ],
    ]);
    const [count, labels] = new BWLabeler().bwlabel(img, dim, 6, true, false);
    // 6-connectivity: the four corners are all disconnected.
    expect(count).toBe(4);
    // Each foreground voxel gets a nonzero label; background stays 0.
    const nonzero = labels.filter((v) => v !== 0).length;
    expect(nonzero).toBe(4);
  });

  it("keeps only the largest cluster per class when requested", () => {
    // One big blob (3 connected voxels) and one small blob (1 voxel).
    const { img, dim } = volume([
      [
        [1, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
      ],
    ]);
    const [, kept] = new BWLabeler().bwlabel(img, dim, 6, true, true);
    const keptCount = kept.filter((v) => v !== 0).length;
    // The isolated single voxel is dropped; the 3-voxel run survives.
    expect(keptCount).toBe(3);
  });

  it("rejects invalid connectivity", () => {
    const { img, dim } = volume([
      [
        [1, 0],
        [0, 1],
      ],
    ]);
    const [count, labels] = new BWLabeler().bwlabel(img, dim, 7, true, false);
    expect(count).toBe(0);
    expect(labels.every((v) => v === 0)).toBe(true);
  });
});
