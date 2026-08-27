import type { ReactNode } from "react";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  message: string;
}

export default function EmptyState({ icon, title, message }: EmptyStateProps) {
  return (
    <div className="state-view">
      {icon && <div className="state-view__icon">{icon}</div>}
      <p className="state-view__title">{title}</p>
      <p className="state-view__message">{message}</p>
    </div>
  );
}
