/** Lightweight pub/sub for decoupling sources, renderer, and UI. */

export function createBus() {
  /** @type {Map<string, Set<Function>>} */
  const handlers = new Map();

  return {
    on(event, fn) {
      if (!handlers.has(event)) handlers.set(event, new Set());
      handlers.get(event).add(fn);
      return () => handlers.get(event)?.delete(fn);
    },
    emit(event, payload) {
      const set = handlers.get(event);
      if (!set) return;
      for (const fn of set) fn(payload);
    },
  };
}
