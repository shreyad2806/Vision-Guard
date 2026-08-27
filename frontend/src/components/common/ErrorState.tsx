import { AlertTriangle } from "lucide-react";

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export default function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
}: ErrorStateProps) {
  return (
    <div className="state-view">
      <AlertTriangle size={40} className="state-view__icon" style={{ color: "var(--red)" }} />
      <p className="state-view__title">{title}</p>
      <p className="state-view__message">{message}</p>
      {onRetry && (
        <button
          type="button"
          className="btn btn--ghost"
          style={{ marginTop: 16 }}
          onClick={onRetry}
        >
          Try Again
        </button>
      )}
    </div>
  );
}
