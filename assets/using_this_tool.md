# Using This Tool

This app helps you asseess skills for your Shop Titans heroes. You can **pick and compare 4-skill builds** for a each **hero class**, using a consistent Rating model that is calculated based on actual simulations and game data.  

Fundamentally it helps answer: ***Is my skill build "good enough", or should I retire or re-roll.***

It helps prepare for the hardest **Legendary** and **Huge** mini-bosses where "chest key" drop rates are highest.

The difference is that this approach rates skills **in the full context of the game** using actual data. 

What this tool ***is not***:  The ratings/percentiles while *speicific*, are *exact* for a one circumstance out of infinite choices.  Meaning: what may be the highest rated build here might not be for a different set of equipment / spirits.  That said the intent is that this tool is "good enough" to enable decisionmaking, and if you want to be precise, use other simulators to validate what works best for you!

---

## Quick Start (30 seconds)

1. **Pick a Hero Class** the Class Summary tab can help inspire ideas.
2. Use **2-Skill Explorer** to find strong “cores”. Shows the ratings of each combination of 3rd/4th skills, and a heat map to visualize.
3. Use **Skill Combo Detail** by seelcting a build, or by setting one manually to see information on each unique 4-skill build.
4. Use **Single Skill Info** to understand whether one particular skill is generally strong (or only strong in certain contexts).

## What “Builds” are in this app

A **build** here means a **unique 4-skill combination** for a single hero class.

- The app de-duplicates by **skill_code** (so you’re looking at each unique 4-skill build once).
- Ratings and charts are **class-specific** (a great skill combo on a Grandmaster can be mediocre on an Arch Druid, etc.).
- This tool uses the Tier 1 names of Skills, in English.
- We only refer to the "souled-up" version of each hero class.

---

# Alpha Testing Notes:

This tool is hosted on a free service, and is in an Alpha test mode. 

## Version Notes:  Known issues / items to address:
1. Tricksters are excluded, as they are well covered for there use with Polonia.
2. Further tuning of the ratings is ongoing. While this will unliekly affect the rating **percentile** of a given build, it may affect the overall rating.
3. There is a long initial load time, and navigation between screens may not work right the first time. Please try again, and it should work right.
4. The simulations take months to execute and process. The data gathered is based on Ancient Jungle Extreme, using equipment / spirits that were current in mid-2025. The ratings have been normalized to extrapolate to current conditions. A refresh of the simulation data will be conducted using Meteor Zone extreme once more T16 blueprints are available (and begin replacing the T15 Best-in-Slot equipment/spirit options).
5. Extreme: Hard Mini-Bosses are difficult for many classes, which causes efficient, defensive (DEF & HP) skills to be highly valued given the sim methodology, and in some cases EVA and Critical Hit skills are less valued. The results follow the data, but the data should be treated as prelimary, and pending the re-run of data noted above. Classes with significant impacts: **Astramancer**, .**Warlock**, **Praetorean** (Blu is undervalued), and **Arch Druid** (high def builds like EWaImpJugMan are undervalued).... <more to come>

## Requests to Alpha users:

PENDING

---

# Details about the Tool

## Notes on Icons / Labels

- **n/q** indicates a Non-Qualifying build in our pipeline (it may have issues like failing constraints).
- Percentiles are always **within the selected class**.

---

## Tab 1 — 2-Skill Explorer

Use this tab to find strong **cores** quickly.

### How to use it
1) Choose **Hero Class**
2) Choose **Skill 1** + **Skill 2**
3) If desired, set filter for the Heat Map, default is to show "Top 10" skills for the class. You can also exclude one ore more skills from the display.
4) A table shows in descending order by rank the best performing Skill 3 + Skill 4.  You can navigate to detials by clicking on the rank number or an individual skill.
5) The heatmap helps you scan “what pairs well with what” at a glance.  You can also click through to a specific build using the heat map.

### Tips
- Start with 1–2 favorite skills as anchors, then explore what they “want” to be paired with.
- There are sometime synergies between skills that can be seen via the heat map.

---

## Tab 2 — Skill Combo Detail

This is the “truth tab.” Use it to explore a specific 4-skill build.

### What you get here
- A build summary card (your selected class + 4 skills)
- **Rating Summary**:
    - Rating for the Build | Percentile rank of the Build among all builds in this Class | Quality Icon | Trend information
    - (1) Ratings are on a 100.0 point scale.  Each class has a max rating (betweeen 99.2 and 100.0).
    - (2) Rank **is the most improtant and accurate aspect of the rating", quality icons show the strength in a visual way.
    - (3) Trend: How does the Build stack up as Quests get harder (AJ Extreme Mini-Bosses).  Upward Graph: growing trend.  Flexed Arm: strong now.  Broken Glass: Sub-optimal
- **Build Rating Table**:
    - Provides insight into the components that contribute to the Rating:  Reliablity, Survival and Skill Efficiency
    - Alpha Note:  This may end up being removed, as it may cause confusion to the average user. feedback appreciated.
- **Component Skill Table:**
    - Skill level summary: Icon, Name and overall quality icon for the skill
    - Sparkline histogram showing how strong the skill is in terms of the quantity of highly rated builds that include this class. The more Green & Yellow, the better!
- **Histogram** mapping *this Build* vs. *all Builds for this Class*
    - For this Class: for each Rating tier (x-axis) the count (y-axis) of Builds.
    - The vertical line shows **this build’s rating** and its **percentile** in-clas.

---

## Tab 3 — Single Skill Info

Use this tab when you want to answer:

> “Is this skill generally strong in this class — or only good in specific combos?”

### What you get here
- Data is for an individual skill for this Class. Use this data carefully - as a skill's true quality is in the 4-skill Build.
- Skill Quality: (1) <.....>
  
### What to look for
- Where the selected skill sits in the class distribution
- Whether it’s consistently strong (many 95th percentile+ Builds), or mostly carried by rare perfect pairings

### When this tab is most useful
- Deciding between two similar skills
- Spotting “trap skills” that look good in isolation but underperform in full builds
- Understanding whether a Tier 1 skill is truly right for **this Build**

---

## Tab 4 - Class Summary

Use this tab to understand:
- The max aassigned rating per class (which is set by performance vs. AJ Extreme mini-bosses)
- The most **Key Skill** for that class based on overall data
- A couple of strong **Example Cores** to help you get started with the 2-Skill Explorer
- Example 4-skill Builds to help you get started with the Skill Combo Detail
- ***Apex Build*** reflects **a top rated build in this model**.  The Tool is not trying to say this is **the best build** for the class.
- ***Example Build 1*** An example of a good/very-good build that relies more on Uncommon skills
- ***Example Build 2*** An example of a good build that relies primarly on Common skills

---

# Disclaimers

This site is a fan-made tool related to the game Shop Titans:
- Not affiliated with Kabam.
- Uses in-game and published data, and provides an external analysis to help players make their own decisions about Hero Skills.
- Uses authorized fan toolkit images
- This site cannot be used to facilitate account sharing, automation or economy manipulation.

# Under the Hood

## Methodology (in plain English)

This tool is built from large batches of simulations using a controlled team and controlled assumptions.

- The Simulation engine mirrors (to within 0.1%) the simulation data in Ress' Hero & Quest Simulator
- To clearly show differences in skills, simulatons were run using a consistent approach:
- (1) Quest: Ancient Jungle Extreme**, no mini-bosses or boosters, Fire Barrier
- (2) 3-hero teams: 2 "control" heroes and the test subject.  For Green and Blue types: control is 1 Chieftains + 1 Conquistador, for Red Types: 2 Conquistadors
- (3) Teset heroes: (a) Level 50, (b) 80 seeds for each trait, (c) use Epic equipment.
- (3) To date the simulations considered over 2.66 **million** unique Builds (each run through 50,000 simulations).
- (4) To reduce random differences, and unlike Ress's tool: the same set of random "rolls" for combat were used for each case.
- The resulting raw data is refined to inform the ratings.

---

## Rating System

Each build gets a mathematically calculated Rating based on three components:

### R1 — Reliability (Win %)
“How often does this hero help the team win the quest?”

- Based on **Quest Success Rate** from our simulations.
- If two builds are close in raw rating, R1 helps confirm which one is consistently winning.
- R1 uses the simulation data

### R2 — Survival (Safety Margin)
“How well does this hero keep the team safe (winning without knockouts)?”

- Uses the **minimum survival outcome** across the heroes in the test party.
- High R2 builds tend to be stable across bad RNG and nasty enemy spikes (AOE and consecutive targeting).
- R2 uses the simulation data

### R3 — Skill Efficiency (Optimization / “Goldilocks”)
“Are these skills a clean, efficient set — or are they doing redundant / wasteful things?”

- R3 assesses the efficiency of the skills in the Build.  
- R3 is meant to adjust for the difference between the **simulation environment** and **performing well in practice**.
- It also helps identify Builds that are good in the long run (as quests get harder) vs. having strength "now".
- R3 is calculated based on the stats for the hero and projects from the test quest to a "real world" case against AJ Extreme Legendary and Huge mini-bosses

> Important: R3 can vary a bit by class (some classes allow slightly >1.0 for peak performance).

---

# Common Pitfalls (and how to avoid them)

## 1) Chasing a single #1 build
- Since the tool is based on simulaton data, there is randomness and variability.  Plus is a degree of judgment (and maybe slight bias) in setting the ratings.
- **Percentile** quality of a Build within a class is the best way to assess a build.
- Verify by using Ress' tool versus you intended quest targets.

**Suggested approach:** pick from the top percentile band and choose the build that matches your risk tolerance.

## 2) Overvaluing “spiky” builds
Some builds can score well but are fragile or inconsistent.

**Signal:** high rating but R3 looks suspiciously weak.

## 3) Treating one class’s meta as universal
A “must-have” skill in one class might be mid-tier in another.

**Rule:** always evaluate inside the class you’re playing.

---

# FAQ

## “Why does the same build feel different in my roster?”
This tool uses a controlled setup. Your real roster can differ by:
- gear/enchant alignment
- hero seeds / level scaling details
- party composition / champions
- quest modifiers and enemy mix

Use this app to identify **strong candidates**, then validate with your actual roster constraints.

## “What should I do if two builds are basically tied?”
Pick the one with:
- better R1/R2 balance, and/or
- skills that synergize with how you actually play (risk vs speed preference)

---

## Version / Scope

This is a **skills-focused** analysis tool. It’s designed to stay readable and fast while supporting large build libraries per class.

If you see something that looks “wrong” (ex: an obviously weak build ranked too high), that’s exactly the kind of feedback we use to tune R3 and improve the model.
