-- Run once in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
create table if not exists public.screenings (
  id           bigint generated always as identity primary key,
  report_id    text        not null,
  created_at   timestamptz not null default now(),
  patient_name text,
  patient_id   text,
  age          integer check (age between 0 and 120),
  sex          text check (sex in ('Female','Male','Other')),
  flagged      boolean     not null,
  severity     text check (severity in ('low','moderate','high'))
);
create index if not exists screenings_created_idx on public.screenings (created_at desc);

-- Row Level Security on with NO policies: the public/anon key can read and write nothing.
-- Only the server, which holds the secret (service-role) key, can access the table.
alter table public.screenings enable row level security;
