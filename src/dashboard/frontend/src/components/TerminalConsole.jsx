import React, { useEffect, useState, useRef } from 'react'

const WS_URL = 'ws://127.0.0.1:8080/ws/console'

export default function TerminalConsole() {
  const [logs, setLogs] = useState([])
  const bottomRef = useRef(null)

  useEffect(() => {
    let ws
    const connect = () => {
      ws = new WebSocket(WS_URL)
      ws.onmessage = (event) => {
        setLogs((prev) => {
          const updated = [...prev, event.data]
          if (updated.length > 50) return updated.slice(updated.length - 50)
          return updated
        })
      }
      ws.onclose = () => {
        setTimeout(connect, 3000)
      }
    }
    connect()
    return () => {
      if (ws) ws.close()
    }
  }, [])

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  return (
    <section className="terminal-console">
      <div className="terminal-header">
        <span className="terminal-title">APEX CONSOLE</span>
        <div className="terminal-controls">
          <span className="dot dot-red"></span>
          <span className="dot dot-yellow"></span>
          <span className="dot dot-green"></span>
        </div>
      </div>
      <div className="terminal-body">
        {logs.map((log, i) => (
          <div key={i} className="terminal-line">{log}</div>
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  )
}
