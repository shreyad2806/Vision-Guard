import { useCallback, useRef, useState } from "react";
import {
  Upload,
  X,
  FileImage,
  Replace,
} from "lucide-react";

interface ImageUploaderProps {
  onImageSelected: (file: File, previewUrl: string) => void;
  onRemove: () => void;
  selectedFile: File | null;
  previewUrl: string | null;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ACCEPTED_EXTENSIONS = ".jpg,.jpeg,.png,.webp";

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1_048_576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1_048_576).toFixed(1)} MB`;
}

export default function ImageUploader({
  onImageSelected,
  onRemove,
  selectedFile,
  previewUrl,
  disabled,
}: ImageUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSelect = useCallback(
    (file: File) => {
      setError(null);
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError(
          `Unsupported file type. Please upload a JPG, PNG, or WEBP image.`,
        );
        return;
      }
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    },
    [onImageSelected],
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndSelect(file);
      // Reset so re-uploading the same file triggers change
      e.target.value = "";
    },
    [validateAndSelect],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) validateAndSelect(file);
    },
    [disabled, validateAndSelect],
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) setDragActive(true);
    },
    [disabled],
  );

  const handleDragLeave = useCallback(() => {
    setDragActive(false);
  }, []);

  // ----- Preview mode -----
  if (selectedFile && previewUrl) {
    return (
      <div className="upload-preview">
        <div className="upload-preview__image-wrap">
          <img
            src={previewUrl}
            alt={selectedFile.name}
            className="upload-preview__image"
          />
        </div>
        <div className="upload-preview__info">
          <div className="upload-preview__meta">
            <FileImage
              size={14}
              style={{
                display: "inline",
                verticalAlign: "middle",
                marginRight: 4,
              }}
            />
            <strong>{selectedFile.name}</strong>
            <span style={{ marginLeft: 8 }}>
              {formatFileSize(selectedFile.size)}
            </span>
          </div>
          <div className="upload-preview__actions">
            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => inputRef.current?.click()}
              disabled={disabled}
            >
              <Replace size={14} />
              Replace
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={onRemove}
              disabled={disabled}
            >
              <X size={14} />
              Remove
            </button>
          </div>
        </div>
        {/* Hidden input for replace */}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          onChange={handleFileChange}
          className="sr-only"
        />
      </div>
    );
  }

  // ----- Empty upload zone -----
  return (
    <div>
      <div
        className={`upload-zone ${dragActive ? "upload-zone--active" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <Upload size={36} className="upload-zone__icon" />
        <p className="upload-zone__title">
          Drag and drop an image here, or click to browse
        </p>
        <p className="upload-zone__subtitle">
          Select an image to analyze its visual quality
        </p>
        <p className="upload-zone__formats">
          Supported formats: JPG, JPEG, PNG, WEBP
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXTENSIONS}
        onChange={handleFileChange}
        className="sr-only"
      />
      {error && (
        <p
          style={{
            color: "var(--red)",
            fontSize: 13,
            marginTop: 8,
            textAlign: "center",
          }}
        >
          {error}
        </p>
      )}
    </div>
  );
}
