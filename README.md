# ai-csv-batch

Run every row of a CSV file through an AI model (Claude, GPT or Gemini) and get the answers back as new
columns: classify, extract fields, or summarize, row by row.

Built for real batches, where things go wrong halfway:

- **Retries.** A rate limit, a server error, a timeout or a network error is tried again with growing
  pauses (or the pause the provider asks for), instead of stopping the run.
- **Resume.** Every answered row is saved as soon as its answer arrives. Run the same command again after a
  crash or a stop, and rows already answered are not sent again. If you change a row's text, the
  instructions or the fields, the rows affected are done again, so an old answer never ends up next to a new
  text. Changing only the provider or the model does not: to redo the file with another model, use a new
  `output_dir`.
- **Fallback.** List more than one provider: when the first one keeps failing on a row, the next one takes
  over. A provider with a wrong key or an unknown model is switched off after its first failure, so it does
  not fail once per row; so is a provider whose account has run out of credit or quota (recognised from the
  providers' known error messages). A request rejected for one row only (too long, blocked by the provider's
  content filter, declined by the model) fails that row and the run goes on.
- **Checked answers.** The model must answer in a fixed JSON shape, and every answer is checked against the
  allowed values before it is accepted. An answer that is cut short, is not valid JSON or has a wrong value is
  retried, and its tokens are still counted in the cost.
- **Cost log.** Every call that reaches the provider, successful or not, is written to `cost_log.csv`:
  provider, model, outcome, tokens, cost and time. You see what a run cost and where it failed. API keys are
  masked if a provider echoes one back in an error message.

## Quick start

Python 3.11 or newer (tested on 3.12).

```bash
pip install -r requirements.txt
cp config.example.toml config.toml
cp .env.example .env          # then put your API key in .env

python run.py --dry-run --limit 10   # offline, simulated answers: checks your setup at no cost
python run.py --limit 10             # 10 real rows: look at the answers and the cost before the full run
python run.py                        # the rest; run it again any time to resume
```

The output is `out/reviews_labeled.csv`: your rows, plus one column per field. Rows not answered yet have
empty cells, and so do rows with no text, which are never sent. The output is written even if the run stops on
an unexpected error, with every answer saved up to that moment.

## The sample data

`data/reviews.csv` holds 100 short product reviews **written for this demo** (no real customers). The
example config asks for three fields per review:

| field | answer |
|---|---|
| `sentiment` | positive, neutral or negative |
| `topic` | quality, delivery, price, support or other |
| `summary` | one sentence, at most 15 words |

## Sample run

The first 10 rows, with the example config (Claude Opus 5.5, effort low), on 13 November 2026:

```
Rows answered this run: 10   already done before: 0   failed: 0
Answered in total: 10 of 100   ->  out/reviews_labeled.csv
Cost of this run: $0.0276   (every call is in out/cost_log.csv)
  anthropic  claude-opus-5-5          ok               10

$ python check.py out/reviews_labeled.csv data/expected.csv
Checked cells: 10   matching: 10
```

A few of the answers:

| id | review | sentiment | topic | summary |
|---|---|---|---|---|
| R003 | Arrived three days late and the box was crushed, but the earbuds work fine. | negative | delivery | Delivery was three days late with a crushed box, but earbuds work fine. |
| R004 | Decent for the price. Nothing special, nothing bad. | neutral | price | Product is decent for its price, neither special nor bad. |
| R010 | Package never arrived and nobody answered my emails for ten days. | negative | delivery | Package never arrived and customer emails went unanswered for ten days. |

About $0.003 a row, so roughly $0.28 for the whole sample file. Prices change: the cost log uses the ones in your
config.

## Your own data

Everything is in `config.toml`: the input file, the id column (it must be unique per row), the column the
model reads, the instructions, and the fields. A field with `choices` must be answered with one of them; a
field without choices is free text.

```toml
[[fields]]
name = "priority"
choices = ["urgent", "normal", "low"]

[[fields]]
name = "product_name"
description = "the product the customer writes about, as written in the text"
```

Providers are tried in the order they are listed. Prices are in USD per million tokens and only feed the
cost log: check them on each provider's pricing page.

## Checking the answers on a sample

Before a large run, label a few rows by hand in a file like `data/expected.csv` (an empty cell is not
checked, useful for free text), run the sample, then:

```bash
python check.py out/reviews_labeled.csv data/expected.csv
```

It prints how many checked cells match, and which ones do not. That is the moment to adjust the
instructions, before paying for the full file.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests run offline, with no key and no cost: retries, fallback, resume after a crash, the cost log, the
answer checks, and each provider adapter fed with the SDK's own response and error types (and, for network
errors, a real SDK client pointed at a closed local port).

## Files

| file | what it does |
|---|---|
| `run.py` | the command: reads the config, runs the batch, writes the output and a summary |
| `batch.py` | the loop: retries, fallback, resume, answer checks, cost log |
| `providers.py` | one small adapter per provider (Anthropic, OpenAI, Gemini) plus an offline mock |
| `check.py` | compares the output with the answers you expect |

## Limits

- One row is one call. For very large files, the providers' batch APIs are cheaper and slower; this tool
  favours seeing results and costs as they come.
- Rows run one after another. That keeps rate limits simple; for big files it can take a while.
- If the run stops in the instant between receiving an answer and saving it, that row is sent again on the
  next run and paid twice. Every other answer is saved and never paid for again, unless you change its text,
  the instructions or the fields.
- Two runs at the same time in the same `output_dir` do not see each other's answers and can pay for the same
  rows twice: run one at a time.
- A provider that is overloaded is retried on every row before the next provider takes over, so a long outage
  of the first provider slows the run down.
- The model's answers are only as good as the instructions and the model. The tool guarantees the format,
  not that every label is right: check a sample.

## License

MIT, see `LICENSE`.
