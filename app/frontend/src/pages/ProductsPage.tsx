import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listProducts, addProduct } from "../api/client";

export default function ProductsPage() {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);

  const { data, isLoading } = useQuery({
    queryKey: ["products"],
    queryFn: listProducts,
  });

  const mutation = useMutation({
    mutationFn: () => addProduct(name, files),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["products"] });
      setName("");
      setFiles([]);
    },
  });

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h2 className="text-xl font-bold">Product Registration</h2>

      {/* Add form */}
      <div className="bg-white border border-[var(--color-border)] rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Product Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter product name"
            className="w-full px-3 py-2 border border-[var(--color-border)] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">
            Images (1-3)
          </label>
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
            className="text-sm"
          />
          {files.length > 0 && (
            <p className="text-xs text-[var(--color-text-muted)] mt-1">
              {files.length} file(s) selected
            </p>
          )}
        </div>
        <button
          onClick={() => mutation.mutate()}
          disabled={!name.trim() || files.length === 0 || mutation.isPending}
          className="px-4 py-2 rounded-lg text-sm font-medium bg-[var(--color-primary)] text-white disabled:opacity-50"
        >
          {mutation.isPending ? "Adding..." : "Add Product"}
        </button>
        {mutation.isError && (
          <p className="text-sm text-[var(--color-danger)]">
            {(mutation.error as Error).message}
          </p>
        )}
        {mutation.isSuccess && (
          <p className="text-sm text-[var(--color-success)]">
            Product added successfully!
          </p>
        )}
      </div>

      {/* Product list */}
      <div className="bg-white border border-[var(--color-border)] rounded-xl p-6">
        <h3 className="font-semibold mb-3">Registered Products</h3>
        {isLoading ? (
          <p className="text-sm text-[var(--color-text-muted)]">Loading...</p>
        ) : data?.products.length === 0 ? (
          <p className="text-sm text-[var(--color-text-muted)]">
            No products registered yet.
          </p>
        ) : (
          <ul className="divide-y divide-[var(--color-border)]">
            {data?.products.map((p) => (
              <li
                key={p.name}
                className="py-2 flex items-center justify-between text-sm"
              >
                <span className="font-medium">{p.name}</span>
                <span className="text-[var(--color-text-muted)]">
                  {p.embedding_count} embeddings
                </span>
              </li>
            ))}
          </ul>
        )}
        {data && (
          <p className="text-xs text-[var(--color-text-muted)] mt-3">
            Total embeddings: {data.total_embeddings}
          </p>
        )}
      </div>
    </div>
  );
}
