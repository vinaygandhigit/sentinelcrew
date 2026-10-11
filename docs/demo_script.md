# Demo video script (about 6 minutes)

**Before recording:** set `LLM_PROVIDER=gemini` and your `GEMINI_API_KEY` in `.env`, run `sentinelcrew check-llm`, delete `var/` for a clean case list, then run `sentinelcrew serve` and open http://127.0.0.1:8000 full screen. Keep a terminal ready in a second window.

Upload the video to YouTube as Public or Unlisted (not Private), and paste the link into the README and the design document.

---

### 0:00 to 0:30. The problem (title slide or your face on camera)

> "A SOC gets thousands of alerts a day and most of them are false alarms. Every one still costs an analyst the same routine: pull the logs, check the IPs, find out whether IT was doing maintenance, decide, contain, and write it up. SentinelCrew is a team of AI agents that does that whole routine, and argues with itself before it acts."

### 0:30 to 1:30. Architecture (show the Mermaid diagram from the README on GitHub)

> "There are seven agents. Triage scores the alert with an Isolation Forest I wrote from scratch, plus expert rules. The Investigator and Threat-Intel agents work in parallel. Case Memory recalls similar past incidents. The Orchestrator merges all of that into one confidence score.
> Then comes the part I'm proudest of: a Critic agent tries to prove the verdict wrong. Every challenge it raises is checked against real data, like the asset inventory, the change calendar, the allowlist, threat intel, and past cases.
> After that the Responder contains the threat. High-risk actions always wait for a human. The Reporter writes the incident report.
> The LLM proposes, and the tools verify. Gemini runs this demo, Claude runs production, and with no key at all it still works."

### 1:30 to 2:45. Scenario 1: brute force leading to compromise

Click **Brute-force -> account compromise**. While the messages stream in, point out:

- Triage: "46 failed logons, success after failures, ML anomaly 1.00. Priority critical."
- Investigator: "It pivots from the VPN logon to the finance workstation the attacker reached next."
- Threat-Intel: "The source is a known Tor exit node."
- Critic: "Is this a user mistyping a password? Refuted: 46 failures from an external IP. Was there a change window? No. Confidence goes **up**, from 0.87 to 0.97."
- Case file: "The IP was blocked and sessions reset automatically. Disabling the account is high-risk, so it waits for me." Click **Approve**.

### 2:45 to 3:45. Scenario 2: the noisy false positive

Click **Noisy false positive: authorised vulnerability scanner**.

> "This is a 140-port scan, and the IDS calls it high severity. Without the Critic it would land in an analyst's queue. Watch what happens."

- Critic: "Is the source an authorised scanner? **Upheld**: SEC-SCAN-01 belongs to IT Security. Was there an approved change? **Upheld**: CHG-2041. The closest past case was a benign scan too."
- Point at the confidence trace: "It falls from 0.58 to 0.03, closes as a false positive, and no human has to look at it."

### 3:45 to 4:45. Scenario 3: beaconing and exfiltration

Click **Malware beaconing + data exfiltration**.

- "Word launched encoded PowerShell, which beacons every 60 seconds with a regularity of 0.98, then rundll32, then 260 MB goes out to a known C2 domain."
- "Memory finds a similar past Cobalt Strike case. The Critic can't find a benign explanation."
- "The domain and IP are sinkholed automatically. Isolating the engineer's laptop is disruptive, so it's pending my approval." Open the **Incident report**, scroll through the evidence and the debate table, and download it.

### 4:45 to 5:30. Measured value and the terminal

In the terminal:

```bash
sentinelcrew evaluate
```

> "This is the same pipeline with and without the Critic. Without it, accuracy is 60% and two false positives get escalated. With it, accuracy is 100%, and the analyst queue drops from four cases to the two real incidents."

Point at the left rail metrics: the auto-close rate, the mean time to verdict in milliseconds, and the pending approvals.

### 5:30 to 6:00. Quality and close

```bash
pytest --cov=src/sentinelcrew
```

> "There are 178 tests at about 99% coverage. They run without network access or API keys, and they cover LLM outages, malformed model output, and attempts to escape the action allow-list. The code is on GitHub with a full technical design document. Thank you."

---

**Tips:** Record at 1080p. Zoom the browser to about 110% so the text is readable. If the Gemini response is slow on the first call, run scenario 1 once before recording so the recording starts warm.

