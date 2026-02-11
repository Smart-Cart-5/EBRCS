interface Props {
  frameBase64: string | null;
}

export default function AnnotatedFrame({ frameBase64 }: Props) {
  if (!frameBase64) {
    return (
      <div className="bg-gray-900 rounded-xl aspect-video flex items-center justify-center">
        <span className="text-gray-400 text-sm">Waiting for frames...</span>
      </div>
    );
  }

  return (
    <img
      src={`data:image/jpeg;base64,${frameBase64}`}
      alt="Annotated checkout frame"
      className="w-full rounded-xl"
    />
  );
}
