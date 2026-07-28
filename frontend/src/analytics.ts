type AnalyticsProperties = Record<string, unknown>

type AnalyticsClient = {
  capture: (eventName: string, properties?: AnalyticsProperties) => unknown
  captureException: (error: unknown, properties?: AnalyticsProperties) => unknown
}

let analyticsClient: AnalyticsClient | null = null

export function setAnalyticsClient(client: AnalyticsClient): void {
  analyticsClient = client
}

export function captureAnalyticsEvent(
  eventName: string,
  properties: AnalyticsProperties,
): boolean {
  if (!analyticsClient) return false
  try {
    analyticsClient.capture(eventName, properties)
    return true
  } catch {
    // Analytics must never interrupt a local game.
    return false
  }
}

export function captureAnalyticsException(
  error: unknown,
  properties: AnalyticsProperties,
): boolean {
  if (!analyticsClient) return false
  try {
    analyticsClient.captureException(error, properties)
    return true
  } catch {
    // Error reporting must never cause a second application error.
    return false
  }
}
