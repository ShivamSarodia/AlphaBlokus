import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/sora'
import './index.css'
import App from './App.tsx'

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
    })
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
