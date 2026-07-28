import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/sora'
import './index.css'
import App from './App.tsx'
import { setAnalyticsClient } from './analytics'

const posthogToken = import.meta.env.VITE_POSTHOG_PROJECT_TOKEN
const posthogHost = import.meta.env.VITE_POSTHOG_HOST || 'https://us.i.posthog.com'

async function renderApp() {
  let app = <App />
  if (posthogToken) {
    const [{ default: posthog }, { PostHogProvider }] = await Promise.all([
      import('posthog-js'),
      import('@posthog/react'),
    ])
    posthog.init(posthogToken, {
      api_host: posthogHost,
      defaults: '2026-05-30',
      autocapture: true,
      capture_pageview: 'history_change',
      capture_pageleave: true,
      capture_exceptions: true,
      disable_session_recording: false,
      enable_recording_console_log: true,
      capture_performance: {
        network_timing: true,
        web_vitals: true,
      },
      session_recording: {
        recordHeaders: true,
        recordBody: true,
        streamNetworkBody: true,
        maskCapturedNetworkRequestFn: (request) => ({
          ...request,
          responseBody: undefined,
        }),
      },
      logs: {
        captureConsoleLogs: true,
        serviceName: 'alphablokus-frontend',
        environment: import.meta.env.MODE,
      },
    })
    setAnalyticsClient(posthog)
    app = (
      <PostHogProvider client={posthog}>
        <App />
      </PostHogProvider>
    )
  }

  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      {app}
    </StrictMode>,
  )
}

void renderApp()
