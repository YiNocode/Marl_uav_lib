/**
 * Local playback clock for file replay (decoupled from live backend).
 */

export class LocalPlayback {
  constructor({ onFrame, onStatus }) {
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.frames = [];
    this.stepDt = 1 / 60;
    this.speed = 0.25;
    this.index = 0;
    this.playing = false;
    this._timer = null;
    this.meta = {};
    this.summary = {};
    this.episode = {};
    this.sceneId = "generic";
  }

  loadDocument(doc) {
    this.stop();
    this.frames = Array.isArray(doc.frames) ? doc.frames : [];
    this.stepDt = Number(doc.step_dt || doc.meta?.step_dt || 1 / 60);
    this.meta = doc.meta || {};
    this.summary = doc.summary || {};
    this.episode = doc.episode || {};
    this.sceneId = doc.scene_id || this.frames[0]?.scene_id || "generic";
    this.index = 0;
    if (this.frames.length) this.onFrame?.(this.frames[0], { replay: true });
    this._emitStatus();
  }

  play() {
    if (!this.frames.length) return;
    this.playing = true;
    this._schedule();
    this._emitStatus();
  }

  pause() {
    this.playing = false;
    if (this._timer) clearTimeout(this._timer);
    this._timer = null;
    this._emitStatus();
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
  }

  stop() {
    this.pause();
    this.index = 0;
    this._emitStatus();
  }

  setSpeed(speed) {
    this.speed = Math.max(Number(speed) || 0.05, 0.01);
    if (this.playing) this._schedule();
    this._emitStatus();
  }

  seek(index) {
    this.index = Math.max(0, Math.min(Number(index) || 0, this.frames.length - 1));
    this.onFrame?.(this.frames[this.index], { replay: true, seek: true });
    this._emitStatus();
  }

  step(delta = 1) {
    this.seek(this.index + delta);
  }

  _schedule() {
    if (this._timer) clearTimeout(this._timer);
    if (!this.playing || !this.frames.length) return;
    const delay = Math.max((this.stepDt / this.speed) * 1000, 5);
    this._timer = setTimeout(() => {
      if (!this.playing) return;
      if (this.index >= this.frames.length - 1) {
        this.pause();
        return;
      }
      this.index += 1;
      this.onFrame?.(this.frames[this.index], { replay: true });
      this._schedule();
      this._emitStatus();
    }, delay);
  }

  _emitStatus() {
    this.onStatus?.({
      mode: "replay",
      playing: this.playing,
      speed: this.speed,
      index: this.index,
      total: this.frames.length,
      episode: this.episode,
      summary: this.summary,
      sceneId: this.sceneId,
    });
  }
}
