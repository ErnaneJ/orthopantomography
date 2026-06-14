const FALLBACK_TZ = "America/Fortaleza";

function getTimezone(): string {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    return tz || FALLBACK_TZ;
  } catch {
    return FALLBACK_TZ;
  }
}

const TZ = getTimezone();

export function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-US", {
      timeZone: TZ,
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

export function formatDateShort(iso: string): string {
  try {
    return new Date(iso).toLocaleString("en-US", {
      timeZone: TZ,
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}
