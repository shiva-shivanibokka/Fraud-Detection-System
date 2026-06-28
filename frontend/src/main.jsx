import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import * as Sentry from "@sentry/react";
import App from "./App.jsx";

// Sentry error tracking — optional. Active only when VITE_SENTRY_DSN is set at
// build time; otherwise this is a no-op and adds no network calls.
const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN;
if (SENTRY_DSN) {
  Sentry.init({
    dsn: SENTRY_DSN,
    environment: import.meta.env.MODE,
    tracesSampleRate: 0.1,
  });
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Sentry.ErrorBoundary fallback={<p style={{ color: "#f87171", padding: 32 }}>
      Something went wrong. The error has been reported.
    </p>}>
      <App />
    </Sentry.ErrorBoundary>
  </StrictMode>
);
