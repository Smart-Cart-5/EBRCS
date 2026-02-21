import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getBilling, getChatbotSuggestions, queryChatbot } from "../api/client";
import { useSessionStore } from "../stores/sessionStore";

type Sender = "user" | "assistant";

interface ChatMessage {
  id: string;
  sender: Sender;
  text: string;
}

interface ChatbotChoice {
  item_no: string;
  product_name: string;
  label: string;
}

type BrowserSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: (() => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  start: () => void;
};

type BrowserSpeechRecognitionCtor = new () => BrowserSpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: BrowserSpeechRecognitionCtor;
    webkitSpeechRecognition?: BrowserSpeechRecognitionCtor;
  }
}

export default function ChatbotWidget() {
  const sessionId = useSessionStore((s) => s.sessionId);
  const billingItems = useSessionStore((s) => s.billingItems);
  const setBilling = useSessionStore((s) => s.setBilling);
  const setBillingState = useSessionStore((s) => s.setBillingState);

  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [choices, setChoices] = useState<ChatbotChoice[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "initial",
      sender: "assistant",
      text: "안녕하세요! 장바구니 금액, 가격, 상품 정보를 물어보세요.",
    },
  ]);

  // --- Drag state ---
  const [pos, setPos] = useState({ x: window.innerWidth - 64, y: 16 });
  const dragging = useRef(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const hasMoved = useRef(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    dragging.current = true;
    hasMoved.current = false;
    dragOffset.current = { x: e.clientX - pos.x, y: e.clientY - pos.y };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [pos]);

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!dragging.current) return;
      hasMoved.current = true;
      const btnSize = 48;
      const nx = Math.max(0, Math.min(window.innerWidth - btnSize, e.clientX - dragOffset.current.x));
      const ny = Math.max(0, Math.min(window.innerHeight - btnSize, e.clientY - dragOffset.current.y));
      setPos({ x: nx, y: ny });
    };
    const onUp = () => { dragging.current = false; };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  const hasCartItems = useMemo(() => Object.keys(billingItems).length > 0, [billingItems]);

  useEffect(() => {
    if (!open) return;
    getChatbotSuggestions(sessionId || undefined)
      .then((res) => setSuggestions(res.suggestions || []))
      .catch(() => setSuggestions([]));
  }, [open, sessionId, hasCartItems]);

  useEffect(() => {
    if (!open) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, open]);

  const sendMessage = async (question: string, displayText?: string) => {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      sender: "user",
      text: displayText || trimmed,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await queryChatbot({
        question: trimmed,
        session_id: sessionId || undefined,
      });

      if (res.cart_update?.billing_items && sessionId) {
        try {
          const latest = await getBilling(sessionId);
          setBillingState(latest);
        } catch {
          setBilling(res.cart_update.billing_items);
        }
      }

      if (res.cart_update?.candidates?.length) {
        setChoices(res.cart_update.candidates);
      } else {
        setChoices([]);
      }

      const assistantMessage: ChatMessage = {
        id: `${Date.now()}-assistant`,
        sender: "assistant",
        text: res.answer,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-error`,
          sender: "assistant",
          text: "답변 생성 중 오류가 발생했어요. 잠시 후 다시 시도해주세요.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendQuestion = async (question: string) => {
    await sendMessage(question);
  };

  const selectChoice = async (choice: ChatbotChoice) => {
    setChoices([]);
    await sendMessage(`__select__:${choice.label}`, `선택: ${choice.product_name}`);
  };

  const handleVoiceInput = () => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) {
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-voice-unsupported`,
          sender: "assistant",
          text: "이 브라우저는 음성 인식을 지원하지 않아요. 텍스트로 질문해주세요.",
        },
      ]);
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.lang = "ko-KR";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setListening(true);
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);
    recognition.onresult = (event) => {
      const transcript = event.results[0]?.[0]?.transcript || "";
      if (transcript.trim()) {
        setInput(transcript.trim());
      }
    };

    recognition.start();
  };

  // Compute panel position so it stays on screen
  const panelW = 360;
  const panelH = 480;
  const panelX = Math.min(pos.x, window.innerWidth - panelW - 8);
  const panelY = pos.y + 56;
  const flipUp = panelY + panelH > window.innerHeight;
  const finalPanelY = flipUp ? Math.max(8, pos.y - panelH - 8) : panelY;

  return (
    <>
      {/* Draggable FAB */}
      <button
        type="button"
        style={{ left: pos.x, top: pos.y }}
        className="fixed z-50 w-12 h-12 rounded-full bg-[var(--color-primary)] hover:bg-[var(--color-primary-hover)] text-white shadow-lg flex items-center justify-center cursor-grab active:cursor-grabbing select-none touch-none"
        aria-label="챗봇 열기"
        onPointerDown={onPointerDown}
        onClick={() => {
          if (!hasMoved.current) setOpen((v) => !v);
        }}
      >
        {open ? "✕" : "💬"}
      </button>

      {/* Chat panel */}
      {open && (
        <div
          style={{ left: Math.max(8, panelX), top: finalPanelY }}
          className="fixed z-50 w-[min(92vw,360px)] bg-[var(--color-surface)] border border-[var(--color-border)] rounded-2xl shadow-xl overflow-hidden"
        >
          <div className="px-4 py-3 border-b border-[var(--color-border)] bg-[var(--color-primary-light)]">
            <p className="text-sm font-semibold text-[var(--color-text)]">스마트 챗봇</p>
            <p className="text-xs text-[var(--color-text-secondary)]">
              {hasCartItems ? "장바구니 기반 답변 준비됨" : "장바구니가 비어있어요"}
            </p>
          </div>

          <div ref={scrollRef} className="h-80 overflow-y-auto p-3 space-y-2 bg-[var(--color-bg)]">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`max-w-[90%] px-3 py-2 rounded-xl text-sm whitespace-pre-wrap ${
                  msg.sender === "user"
                    ? "ml-auto bg-[var(--color-primary)] text-white"
                    : "bg-white text-[var(--color-text)] border border-[var(--color-border)]"
                }`}
              >
                {msg.text}
              </div>
            ))}
            {loading && (
              <div className="inline-flex px-3 py-2 rounded-xl text-sm bg-white text-[var(--color-text-secondary)] border border-[var(--color-border)]">
                답변 생성 중...
              </div>
            )}
          </div>

          {suggestions.length > 0 && (
            <div className="px-3 py-2 border-t border-[var(--color-border)] flex flex-wrap gap-2">
              {suggestions.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => sendQuestion(q)}
                  className="px-2 py-1 rounded-full text-xs bg-[var(--color-primary-light)] text-[var(--color-primary)] hover:opacity-90"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {choices.length > 0 && (
            <div className="px-3 py-2 border-t border-[var(--color-border)] bg-white">
              <p className="text-xs text-[var(--color-text-secondary)] mb-2">어떤 상품을 선택할까요?</p>
              <div className="flex flex-wrap gap-2">
                {choices.map((choice) => (
                  <button
                    key={choice.label}
                    type="button"
                    onClick={() => selectChoice(choice)}
                    className="px-2 py-1 rounded-lg text-xs border border-[var(--color-border)] hover:bg-[var(--color-primary-light)]"
                  >
                    {choice.product_name}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="p-3 border-t border-[var(--color-border)] flex items-center gap-2">
            <button
              type="button"
              onClick={handleVoiceInput}
              className={`w-9 h-9 rounded-full border border-[var(--color-border)] flex items-center justify-center ${
                listening ? "bg-[var(--color-primary-light)] text-[var(--color-primary)]" : "bg-white text-[var(--color-text)]"
              }`}
              aria-label="음성 인식"
            >
              🎤
            </button>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") sendQuestion(input);
              }}
              placeholder="질문을 입력하세요"
              className="flex-1 h-9 px-3 rounded-lg border border-[var(--color-border)] bg-white text-sm outline-none focus:border-[var(--color-primary)]"
            />
            <button
              type="button"
              onClick={() => sendQuestion(input)}
              disabled={loading || !input.trim()}
              className="h-9 px-3 rounded-lg bg-[var(--color-primary)] disabled:opacity-50 text-white text-sm"
            >
              전송
            </button>
          </div>
        </div>
      )}
    </>
  );
}
