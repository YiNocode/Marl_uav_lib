/** Agent trajectory buffers keyed by label. */

export function createTrailStore(maxLen = 400) {
  /** @type {Map<string, number[][]>} */
  const trails = new Map();

  return {
    clear() {
      trails.clear();
    },
    onFrame(frame) {
      if (frame.event === "episode_start") {
        trails.clear();
        return;
      }
      if (!frame.positions) return;
      frame.positions.forEach((pos, i) => {
        const key = frame.agent_labels?.[i] || `A${i}`;
        if (!trails.has(key)) trails.set(key, []);
        const t = trails.get(key);
        t.push([pos[0], pos[1]]);
        if (t.length > maxLen) t.shift();
      });
    },
    get(label) {
      return trails.get(label) || [];
    },
    entries() {
      return trails.entries();
    },
  };
}
