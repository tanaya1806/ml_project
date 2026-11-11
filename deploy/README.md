# Deploying NeuroCare (Supabase + Hugging Face Spaces, free tiers)

Records go to **Supabase Postgres**; the app (Python server + trained models) runs in a **Hugging Face Space (Docker)**.
Only the screening outcome and the entered patient details are stored, never the uploaded scan. The Supabase key stays on the server.

## 1. Supabase (database)
1. supabase.com -> New project (free plan). Pick any region and a database password you will not need again.
2. SQL Editor -> New query -> paste `deploy/supabase_schema.sql` -> Run. (Creates the `screenings` table with Row Level Security on and no public policies.)
3. Project Settings -> API: copy the **Project URL** and the **secret / service_role key**. Never put the secret key in the front end or in git.

## 2. Hugging Face Space (app)
1. huggingface.co -> New Space -> SDK **Docker** -> hardware *CPU basic (free)* -> visibility **Private** (recommended).
2. In the Space: Settings -> Variables and secrets -> add three **Secrets**:
   - `SUPABASE_URL` = the Project URL
   - `SUPABASE_SERVICE_KEY` = the secret key
   - `APP_ACCESS_CODE` = a code your staff type on the sign-in screen
3. From the project root:
   ```bash
   ./deploy/build_space.sh                       # creates space_build/ (about 1 MB)
   git clone https://huggingface.co/spaces/<you>/<space-name> hf_space
   cp -R space_build/. hf_space/ && cd hf_space
   git add -A && git commit -m "NeuroCare screening app" && git push
   ```
   The Space builds the Docker image (a few minutes, PyTorch CPU) and then serves the app on its URL.

## Run locally against Supabase (optional)
```bash
SUPABASE_URL=... SUPABASE_SERVICE_KEY=... APP_ACCESS_CODE=... python3 testing_data/app/server.py
```
Without the Supabase variables the app uses a local SQLite file (`testing_data/app/records.db`); without `APP_ACCESS_CODE` there is no sign-in.

## Limits to know
- Free Spaces go to sleep after a period of inactivity; the first visit afterwards takes a minute to wake. Free Supabase projects pause after about a week without use (resume in the dashboard).
- Free-tier limits change; check both sites' current pricing pages.
- This is a research prototype, not a medical device. Use aliases instead of real patient names, keep the Space private, and do not enter real patient data without consent and approval.
