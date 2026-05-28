/** Scene plugin registry — add new experiment visualizations here. */

/** @type {Map<string, object>} */
const scenes = new Map();

export function registerScene(scene) {
  if (!scene?.id) throw new Error("scene.id required");
  scenes.set(scene.id, scene);
}

export function getScene(sceneId) {
  return scenes.get(sceneId) || scenes.get("generic") || [...scenes.values()][0];
}

export function resolveSceneId(frame) {
  return frame?.scene_id || "pursuit_3v1";
}

export function listScenes() {
  return [...scenes.keys()];
}
