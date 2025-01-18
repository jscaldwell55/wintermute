import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async () => {
    if (!query.trim()) {
      alert("Please enter a query.");
      return;
    }

    try {
      const res = await axios.post("https://your-backend-url.vercel.app/query", {
        prompt: query,
        top_k: 5, // Modify if needed
      });
      setResponse(res.data.response || "No response received.");
    } catch (error) {
      console.error("Error fetching data:", error);
      setResponse("An error occurred while fetching data.");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Memory System Interface</h1>
        <textarea
          placeholder="Type your query here..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleSubmit}>Submit Query</button>
        <div>
          <h2>Response:</h2>
          <p>{response}</p>
        </div>
      </header>
    </div>
  );
}

export default App;
