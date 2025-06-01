"use client"

import type React from "react"

import { useState } from "react"
import { Send, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface ConversationalSearchProps {
  onResults: (results: any[]) => void
}

export default function ConversationalSearch({ onResults }: ConversationalSearchProps) {
  const [query, setQuery] = useState("")
  const [conversation, setConversation] = useState<
    {
      type: "user" | "assistant" | "thinking"
      content: string
    }[]
  >([])
  const [loading, setLoading] = useState(false)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    console.log("query before fetch:", query);

    setLoading(true)
  {
      try {
      setLoading(true)
      console.log("query before fetch:", query);
      const response = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      })

      if (!response.ok) {
        throw new Error("API error")
      }

      const data = await response.json()

      // Update assistant response
      setConversation((prev) => [
        { type: "assistant", content: data.response },
      ])

      onResults(data.products)

      setQuery("")
      setLoading(false)
    } catch (error) {
      console.error("Error fetching response:", error)
      setConversation((prev) => [
        {
          type: "assistant",
          content: "I'm sorry, I encountered an error. Please try again.",
        },
      ])
      setLoading(false)
    }
  }
  }

  return (
    <div className="w-full">
      <form onSubmit={handleSearch} className="flex items-center space-x-2 mb-4">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about products or describe what you're looking for..."
          className="flex-1"
          disabled={loading}
        />
        <Button type="submit" size="icon" disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </Button>
      </form>

      {conversation.length > 0 && (
        <div className="space-y-3 mb-4">
          {conversation.map((message, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg whitespace-pre-line ${
                message.type === "user"
                  ? "bg-primary/10 ml-auto max-w-[80%]"
                  : "bg-gray-100 mr-auto max-w-[80%]"
              }`}
            >
              {message.content}
            </div>
          ))}
          {loading && (
            <div className="bg-gray-100 p-3 rounded-lg flex items-center space-x-2 mr-auto max-w-[80%]">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Thinking...</span>
            </div>
          )}
        </div>
      )}

    </div>
  )
}
