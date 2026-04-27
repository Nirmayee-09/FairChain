# FairChain Demo Video Script (3 Minutes)

## Setup & Tech Requirements
- **Resolution:** 1080p60
- **Recording Tool:** OBS Studio
- **Browser:** Chrome (Fullscreen, no extensions visible, hardware acceleration ON to prevent Mapbox lag)
- **Presenters:** Garud (Voiceover), Nirmayee (Screen Driver)

---

## [0:00 - 0:30] The Problem & Vision
**Visual:** 
- Start on Title Slide (FairChain: Fair & Predictive Supply Chain Intelligence).
- Cut to Slide 2: High-impact imagery of flooded roads and logistics bottleneck.

**Voiceover (Garud):**
"Every year, unpredicted extreme weather events cost the Indian supply chain billions in holding costs and spoilage. Current logistics software reacts too slowly—and worse, it is inherently biased against local SME transporters, favoring massive enterprise carriers just because they have more historical data. Today, we're introducing FairChain: predictive, fair, and fast logistics routing."

---

## [0:30 - 1:30] The Technical Stack & Predictive Anomaly (T-4 Hours)
**Visual:**
- Cut to the FairChain Next.js Dashboard (Mapbox layer showing Chennai region).
- The timeline slider moves to 'T-8 Hours'. A mild yellow warning appears over NH48.
- Timeline slider moves to 'T-4 Hours'. The segment turns bright red. An alert modal pops up.

**Voiceover (Garud):**
"Here is the Chennai Flood 2023 scenario. At T-4 hours—four full hours before the official road closure—our Isolation Forest machine learning models process live IMD rainfall spikes and telematics data. Our system flags a critical disruption."

**Visual:**
- Nirmayee clicks the red segment. A Gemini-powered impact summary opens.

**Voiceover (Garud):**
"To translate raw ML scores into human action, we integrated the Google Gemini API. Here, it instantly generates a human impact summary: 'Severe flooding puts 50 workers and ₹20L in perishable cargo at immediate risk,' along with one-sentence actionable advice to reroute immediately."

---

## [1:30 - 2:30] The 'Aha!' Moment: Fairness in Action
**Visual:**
- Nirmayee clicks 'Generate Alternative Routes'. 
- Two options appear: Route A (Enterprise Partner - ₹5.0L) and Route B (Local SME - ₹1.5L).
- The raw ML confidence for Route A is 0.95, and Route B is 0.45.
- Nirmayee toggles the 'Fairness Scorecard' view.

**Voiceover (Garud):**
"Here is the 'Aha!' moment. When finding a new route, the traditional AI suggests an expensive Enterprise Partner over a highly capable Local SME. Why? Simply because the enterprise has driven the route more times, creating a data volume bias."

**Visual:**
- The Dashboard debiases the score. Route B's fairness-adjusted score becomes 0.92.
- A green checkmark appears next to the Local SME.

**Voiceover (Garud):**
"By applying our fairness algorithms, we normalize the confidence scores based on vendor size. We instantly debias the AI, preventing a monopoly by enterprise carriers during crisis events and securing a 22% cost reduction by safely utilizing qualified local SMEs."

---

## [2:30 - 3:00] Conclusion & Future
**Visual:**
- Cut to the final Pitch Deck slide (Open Innovation & Roadmap).
- Fade to FairChain Logo and GitHub QR code.

**Voiceover (Garud):**
"FairChain isn't just about surviving disruptions; it's about equitable growth. By combining predictive anomaly detection, Google Gemini explainability, and fairness-aware routing, we make logistics resilient and fair for everyone. Thank you."
