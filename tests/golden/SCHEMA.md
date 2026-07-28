
# Golden Test Case Schema (v1.0)

> Every file under `tests/golden/**/*.json` MUST conform to this schema.
> The schema is **strict**; any deviation fails CI.

## Top-level structure

```json
{
  "schema_version": "1.0",
  "rules_version": "1.0.0",
  "category": "battle | move_gen | siling_flag | stronghold | inference | setup_validation | full_game",
  "id": "string; unique within category; e.g. 'siling_vs_bomb_src'",
  "description": "human-readable intent; 1-3 sentences",
  "tags": ["optional", "labels", "like", "Q7", "belief-inference"],
  "legacy_divergence": null,        // or string explaining why this case diverges from legacy engine
  "debug_include_private": false,   // if true, `expected` may include private fields (types, beliefs)

  // Exactly ONE of these initial-state sections:
  "setup_only": { ... },            // for setup_validation cases
  "pre_state": { ... },             // full board state pre-action (for combat/movement/inference)
  "full_setup": { ... },            // 4 lineups + 0 moves (for full_game root)

  // Action being tested:
  "actions": [ { ... }, ... ],      // one or more actions to apply sequentially

  // Expected observable result:
  "expected": {
    "results": [ { ... }, ... ],    // one MoveResult per action
    "final_state": { ... },         // optional; asserts specific fields of GameState
    "terminated": false,
    "winner_team": null,            // 0 (HOME+OPPS), 1 (RIGHT+LEFT), or null
    "draw": false
  }
}
```

## `pre_state` structure

```json
{
  "turn": 0,                        // int seat 0..3
  "move_counter": 0,
  "moves_since_last_combat": 0,
  "pieces": [
    // Each piece: full type + position (world frame)
    { "seat": 0, "index": 3, "type": "SILING",  "pos": [8, 11], "dead": false },
    { "seat": 2, "index": 5, "type": "ZHADAN",  "pos": [8, 5],  "dead": false },
    // ...
  ],
  "info": [
    { "seat": 0, "dead": false, "flag_revealed": false, "jump_count": 0 },
    { "seat": 1, "dead": false, "flag_revealed": false, "jump_count": 0 },
    { "seat": 2, "dead": false, "flag_revealed": false, "jump_count": 0 },
    { "seat": 3, "dead": false, "flag_revealed": false, "jump_count": 0 }
  ]
}
```

## `actions` element

```json
{
  "seat": 0,                        // which seat is acting (must match turn)
  "src": [8, 11],                   // world-frame (x, y)
  "dst": [8, 10]
}
```

## `results` element (expected MoveResult for each action)

```json
{
  "src": [8, 11],
  "dst": [8, 10],
  "event": "MOVE | EAT | KILLED | BOMB",
  "flag_reveal_src": false,
  "flag_reveal_dst": false,
  "flag_captured": false
}
```

## `setup_only` structure (for C1-C5 validation tests)

```json
{
  "lineups": [
    // 30 pieces per seat in index order 0..29; use "NONE" for camps
    ["GONGB", "PAIZH", "LIANZH", "SILING", "YINGZH", ... 30 entries ...],
    [...], [...], [...]
  ],
  "expect_valid": true,             // or false if this setup should be rejected
  "expect_violations": []           // or list of ["C1", "C3"] for which constraints fail
}
```

## `full_setup` structure (for full_game root)

Same as `setup_only.lineups` but always `expect_valid: true` implicit.

## Conventions

- **Coordinates are world-frame** `(x, y)` — never canonical. Rotation tests have a separate schema.
- **Piece types use enum NAMES** (strings, UPPERCASE), never numeric values.
- **Unused fields may be omitted**; validator fills with schema defaults.
- **Binary numeric equality** required for positions; piece counts; move counters.
- **Boolean fields** must be explicit `true`/`false`, not 0/1.

## Validator

`tests/validate_golden_schema.py` is run on every test file at CI start. Malformed JSON halts CI before tests begin.
