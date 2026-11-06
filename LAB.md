# The lab: experiments 1-4 on a list of AstroPT checkpoints, unattended

## Quick start

```
export ASTROPT_PY=/path/to/astroPT/.venv/bin/python   # see README.md
cp lab_config.example.toml my_config.toml        # list your checkpoints; trim what to run (see "Time and usage")
python3 lab.py start my_config.toml              # add --survive-reboot to restart after a reboot as well
```

Then watch **`lab_runs/<name>/STATUS.md`**, the one file with everything: what each lane is doing right now, every
checkpoint, every result so far, what needs attention, and the latest events. The lab finishes by itself. It deletes
the converted weights, writes `results.json`, and exits.

| command | what it does |
|---|---|
| `python3 lab.py status [my_config.toml]` | print STATUS.md |
| `python3 lab.py wake [my_config.toml]` | stop waiting (for a usage limit or a retry) and try again now, e.g. after you reset a limit |
| `python3 lab.py stop [my_config.toml]` | stop cleanly: every job saves its place. `start` again resumes where it stopped |
| `python3 lab.py run my_config.toml` | run in the foreground instead (what `start` runs under its supervisor) |

## What happens to each checkpoint

1. **Prepare** (`ckpt_prepare.py`, on the GPU, about 25 s for 015M and several minutes for 850M):
   - It converts the checkpoint to a numbers-only `weights.bin`.
   - It checks the code in the prompt and our GPU network against the official astroPT model on held-out galaxies.
   - It measures the facts everything is scored against: what each head attends to, the 40 experiment-2 moments, the
     physics score before editing and that of a random network, and how trained the checkpoint looks.

   Any next-patch (autoregressive) astroPT checkpoint works, of any size and at any step, with the `aim` tokeniser
   (MLP encoder and decoder) or the `affine` one (one linear layer each way). That includes checkpoints saved without
   a modality registry. Masked-autoencoder (`mae`) checkpoints, and anything else the
   experiments don't describe, are marked unsupported with the reason and skipped. Checkpoints can come from local
   folders, globs, or Hugging Face model or dataset repos (`hf://...`, or a pasted `https://huggingface.co/...` link).
   `max_per_folder` keeps an evenly spread subset of each folder. Downloads are deleted once converted
   (`resources.keep_downloads`).
2. **Run the jobs** in lanes:
   - one lane per account (claude, codex), so each account only ever has one conversation going;
   - one lane for the GPU (preparing and scoring), so only one GPU job runs at a time.

   Checkpoints are worked through in order. At most `prepare_ahead` checkpoints are prepared ahead of the agents, and
   only when the disk has room.
3. **Tidy up**: once all its jobs are done, its converted weights are deleted (every hard link). Your checkpoint files
   are never touched.

Every agent job runs in the blog-style harness (`harness.py`):
- **No tools:** the agent's own tools are all switched off, and the system prompt is "Answer the user's request."
- **Code:** it runs code by replying with one Python block. Each block runs in a fresh, sealed container: no network,
  your GPU, 600 s, 12 GB of RAM, output cut at 20,000 characters. The container sees only `weights.bin` and the job's
  data.
- **Clean context:** the conversation is resumed turn by turn, and the agent sees nothing on your computer.

The models are Claude Fable 5.1 (`claude`) and Astra, `gpt-6-astra` (`codex`).

## The experiments and their scores

- **Experiment 1: read the weights** without running the network.
  - Prompts are `bare` (one sentence) or `full` (architecture, exact code and rules; with `blind = true` it never says
    how far training has got).
  - **Judge:** both agents grade every answer against the measured facts. For bare answers, 13 items: format,
    transformer, sizes, image patches, galaxies, spiral, how trained it is. For full answers, one item per head plus
    how trained it is. Judges also count *confabulations*: confident claims the measurements contradict.
- **Experiment 2: explain the generation.** Running the network is allowed here.
  - **Explainer:** it studies the network (with 100 real galaxies), writes an account plus testable claims, then
    describes 40 moments (10 galaxies × inner, middle, outer and edge rings) in at most 80 words each, with no digits.
  - **Readers:** they get only the words and the grid position. They draw the patch they expect and say which earlier
    patches it depends on.
  - **Scores:**
    - *patch similarity*: cosine, 0 = no information, 1 = exact;
    - *dependence overlap*: 0 to 1;
    - FVE, the fraction of variance explained, as natural language autoencoders score their explanations.
  - **Conditions:** `desc` (the descriptions), `account` (only the general account), `none` (the baseline) and
    `shuffled` (descriptions of the wrong moments). Every reader reads every explainer.
- **Experiment 3: how explanations change during training.** This is the sweep itself. Give a training run's
  checkpoints with `blind = true`, and every score above is measured against that checkpoint's own behaviour.
- **Experiment 4: edit the weights** to raise the *physics score*.
  - **The score:** the mean R² of eight linear readouts from astroPT's embedding, refitted every time: six physical
    properties (redshift, stellar mass, star-formation rate, size, two colours) and two Galaxy Zoo morphology
    measures (the featured-or-disc and merger vote fractions). No forward passes, no training, and no rescaling a
    whole tensor.
  - **Levels:** `L0` gets one code run, `L1` gets code runs, and `L2` also gets `EVALUATE` calls on validation
    galaxies.
  - **Scoring:** on 1,000 test galaxies, against a random edit of the same size. An edit counts only if the next-patch
    loss rises by less than 2%.

## When things go wrong

- **Usage limit:** the CLI's own message says when the limit resets. That account's lane pauses until then while the
  other lane carries on. `wake` retries at once.
- **Over the output cap:** a max-effort Claude turn that exceeds the 128,000-token output cap is retried once at `xhigh`,
  and that is recorded in its result and in STATUS.md. If it still fails, the job is marked failed at once and the lab
  moves on.
- **Any other failure:** it is retried after 5 minutes, 30 minutes and 2 hours, then marked failed in "Needs attention"
  with the path of its log. Jobs that need it are skipped.
- **The lab process dies:** the supervisor restarts it, and every job resumes where it stopped. A job whose old process
  is still running is adopted: nothing runs twice, and that lane stays busy until it finishes.
- **A reboot** (with `--survive-reboot`): a crontab `@reboot` line starts the lab again. It is removed when the lab
  finishes.

## Files: `lab_runs/<name>/`

- `STATUS.md`: the live status (also `events.log`).
- `results.json`: every job's result, written at the end.
- `state.json`: the lab's own record. Don't edit it while the lab runs.
- `checkpoints/<id>/`: each prepared checkpoint: `meta.json`, `layout.json`, and `truth/` (attention, head facts,
  moments, physics before editing).
- `jobs/<job>/`:
  - `job.json`: exactly what the agent was given;
  - `console.log`;
  - `run/`: every message and reply in `turns/tNNN/`, every code run with its full output in `tool_calls.jsonl`, each
    part's answer, and `final_answer.md`;
  - `run/work`: links to the sandbox folder with the saved conversation (and for experiment 4, `edits/`).

## Time and usage

At the default settings one checkpoint is 34 agent conversations:
- **At max effort:** 4 for experiment 1, 2 explainers and 6 experiment-4 runs.
- **At high effort:** 14 readers and 8 judgings.

Expect very roughly 6 to 20 hours per checkpoint with the two accounts running side by side, longer when usage limits
pause a lane. For a sweep, trim:
- `exp1.prompts = ["full"]`;
- fewer exp2 `conditions` (keep `desc` and `none`);
- `judge.judges` with one judge;
- `exp4.only` on the last checkpoints only.

## Things to know

- **The physics score includes morphology because the physical properties alone barely depend on training.** For
  015M on the test galaxies, the random starting network scores 0.401 and the trained one 0.449. The six physical
  properties alone give 0.492 against 0.506, since colours and stellar mass read out of random features about as well.
  The two morphology measures give 0.128 against 0.275. Results report the physical and morphology parts separately
  too, and STATUS.md shows each checkpoint's random-start score beside its experiment-4 results.
- **FVE is unstable for an untrained network.** Its predictions hardly vary between moments, so FVE can go far below
  zero. Patch similarity and dependence overlap stay meaningful, so read those first.
- **Attention is not dependence.** An untrained network spreads its attention evenly, yet its prediction depends 84% on
  the current patch, through the residual connection. Experiment 2 scores what the prediction depends on (blanking out
  each earlier patch), not where the attention goes.
- **What was tested** (all passed):
  - two checkpoints (a random start and the trained 015M) with both agents, every experiment, tiny budgets, low effort;
  - a simulated usage limit on both lanes, and `wake`;
  - `kill -9` of the lab, with restart and adoption of the jobs still running;
  - `stop` and `start` twice;
  - a clean finish: no processes, containers or weights left;
  - folders, globs and `hf://` sources;
  - the per-experiment `only` filters.
