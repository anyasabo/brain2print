// Mesh helper for the niivue v1 API. v1 dropped NVMeshUtilities.createMZ3 and
// nv.loadFromArrayBuffer, so computed meshes are added as Files via nv.addMesh.

// Encode triangle geometry as a Wavefront OBJ. niivue v1 reads OBJ natively, so
// the result can be wrapped in a File and passed to addMesh.
export function positionsIndicesToObj(
  positions: Float32Array | number[],
  indices: Uint32Array | number[],
): string {
  const lines: string[] = [];
  for (let i = 0; i < positions.length; i += 3) {
    lines.push(`v ${positions[i]} ${positions[i + 1]} ${positions[i + 2]}`);
  }
  // OBJ face indices are 1-based.
  for (let i = 0; i < indices.length; i += 3) {
    lines.push(
      `f ${indices[i] + 1} ${indices[i + 1] + 1} ${indices[i + 2] + 1}`,
    );
  }
  return lines.join("\n");
}
