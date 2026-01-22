import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from "recharts";

type AudioFeatures = {
  valence: number;
  energy: number;
  tempo: number;
};

type Response = {
  emotions: Record<string, number>;
  audio_features: AudioFeatures;
};

export const MoodForm: React.FC = () => {
  const [text, setText] = useState("");
  const [response, setResponse] = useState<Response | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const res = await fetch("http://127.0.0.1:8000/mood", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data: Response = await res.json();
    setResponse(data);
  };

  // Prepare data for charts
  const emotionData = response
    ? Object.entries(response.emotions).map(([key, value]) => ({ emotion: key, value }))
    : [];

  const audioData = response
    ? [
        { feature: "valence", value: response.audio_features.valence },
        { feature: "energy", value: response.audio_features.energy },
        { feature: "tempo", value: response.audio_features.tempo / 200 } // normalize tempo for radar chart
      ]
    : [];

  return (
    <div style={{ maxWidth: 600, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="How are you feeling today?"
          value={text}
          onChange={(e) => setText(e.target.value)}
          style={{ width: "100%", padding: "0.5rem" }}
        />
        <button type="submit" style={{ marginTop: "1rem", padding: "0.5rem 1rem" }}>
          Analyze Mood
        </button>
      </form>

      {response && (
        <>
          <h3 style={{ marginTop: "2rem" }}>Emotion Distribution</h3>
          <BarChart width={500} height={300} data={emotionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="emotion" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>

            <h3 style={{ marginTop: "2rem" }}>Audio Features</h3>
        <RadarChart outerRadius={90} width={400} height={300} data={audioData}>
            <PolarGrid />
            {/* @ts-ignore */}
            <PolarAngleAxis dataKey="feature" {...({} as any)} />
            {/* @ts-ignore */}
            <PolarRadiusAxis angle={30} domain={[0, 1]} {...({} as any)} />
            <Radar
                name="Audio Features"
                dataKey="value"
                stroke="#82ca9d"
                fill="#82ca9d"
                fillOpacity={0.6}
            />
        </RadarChart>


          <h3>Suggested Playlist (Mock)</h3>
          <ul>
            <li>Track A (Valence ~{response.audio_features.valence.toFixed(2)})</li>
            <li>Track B (Energy ~{response.audio_features.energy.toFixed(2)})</li>
            <li>Track C (Tempo ~{response.audio_features.tempo.toFixed(0)} BPM)</li>
          </ul>
        </>
      )}
    </div>
  );
};
