export function formatDuration(seconds: number): string {
  seconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function formatTag(username: string, discriminator: string): string {
  return `${username}#${discriminator}`;
}

export function formatTime(seconds: number): string {
  const s = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
