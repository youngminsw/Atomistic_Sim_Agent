export type ParsedArgs = {
  readonly positionals: readonly string[]
  readonly options: ReadonlyMap<string, string>
  readonly flags: ReadonlySet<string>
}

export class CliArgsError extends Error {
  readonly code: string

  constructor(code: string) {
    super(code)
    this.name = "CliArgsError"
    this.code = code
  }
}

export function parseArgs(argv: readonly string[]): ParsedArgs {
  const positionals: string[] = []
  const options = new Map<string, string>()
  const flags = new Set<string>()
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index]
    if (item === undefined) {
      continue
    }
    if (!item.startsWith("--")) {
      positionals.push(item)
      continue
    }
    const key = item.slice(2)
    const next = argv[index + 1]
    if (next === undefined || next.startsWith("--")) {
      flags.add(key)
      continue
    }
    options.set(key, next)
    index += 1
  }
  return { positionals, options, flags }
}

export function requiredOption(args: ParsedArgs, key: string): string {
  const value = args.options.get(key)
  if (value === undefined || value.length === 0) {
    throw new CliArgsError(`missing_option:${key}`)
  }
  return value
}

export function optionalOption(args: ParsedArgs, key: string): string | undefined {
  const value = args.options.get(key)
  if (value === undefined || value.length === 0) {
    return undefined
  }
  return value
}

export function hasFlag(args: ParsedArgs, key: string): boolean {
  return args.flags.has(key)
}
