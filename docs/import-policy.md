# Import Policy

This repository is primarily an architecture archive. The word "archive" has a precise meaning here: it may include upstream references, minimal clean-room reproductions, annotated copies when redistribution is permitted, and local experiments derived from public architecture descriptions.

It should not become an unlicensed dump of third-party source code, checkpoints, tokenizer files, training data, or model cards.

## Import modes

### 1. Reference-only

Use this mode when the upstream license is unclear, restrictive, mixed, or when the repository is mainly being tracked as a research target.

Allowed content:

- upstream URL;
- paper URL;
- architecture summary;
- module diagram;
- implementation notes;
- local clean-room pseudocode;
- benchmark plan.

Do not copy upstream source code.

### 2. Submodule mirror

Use this mode when preserving upstream history and license is preferable.

Allowed content:

- Git submodule pointing to the upstream repository;
- local README explaining why the upstream is relevant;
- notes and wrappers outside the submodule.

This is usually safer than copying files directly.

### 3. Vendored source import

Use this mode only when redistribution is permitted.

Required:

- preserve upstream license file;
- preserve copyright notices;
- record exact commit SHA;
- record import date;
- record local modifications;
- place the code under `vendor/<family>/<project>/upstream/`.

### 4. Clean-room minimal implementation

Use this mode for local educational implementations.

Required:

- implement from papers, public descriptions, and independent reasoning;
- avoid copying upstream implementation structure line-by-line;
- keep the implementation small and readable;
- document the conceptual source.

Clean-room implementations in this repository are Apache-2.0 unless otherwise stated.

## What not to mirror by default

- model checkpoints;
- tokenizer binary assets;
- training datasets;
- evaluation datasets with redistribution restrictions;
- private or leaked model code;
- code governed by non-commercial or field-of-use restrictions unless isolated and clearly marked.

## Minimal import record

Every import must have a record containing:

```yaml
id:
family:
upstream:
paper:
license:
import_mode:
upstream_commit:
import_date:
local_path:
notes:
```

## Practical repository policy

For fast research, default to reference-only plus clean-room notes. Move to submodule or vendored import only after checking the license.
