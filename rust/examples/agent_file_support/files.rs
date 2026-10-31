use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use walkdir::WalkDir;

const MAX_FILE: usize = 1_048_576;
const MAX_OUTPUT: usize = 65_536;
const MAX_ENTRIES: usize = 1_000;
const MAX_VISITS: usize = 20_000;

#[derive(Clone)]
pub struct Files {
    root: PathBuf,
    excluded: Vec<PathBuf>,
}
impl Files {
    pub fn new(root: &Path) -> Result<Self> {
        let root = root.canonicalize().context("resolve repository root")?;
        ensure!(root.is_dir(), "repository root is not a directory");
        Ok(Self {
            root,
            excluded: Vec::new(),
        })
    }
    pub fn excluding(mut self, path: &Path) -> Self {
        self.excluded.push(path.to_owned());
        self
    }
    fn is_excluded(&self, path: &Path) -> bool {
        self.excluded.iter().any(|excluded| {
            path == excluded
                || (path.parent() == excluded.parent()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(".lockstep-cache-")))
        })
    }
    fn resolve(&self, relative: &str) -> Result<PathBuf> {
        let relative = Path::new(relative);
        ensure!(!relative.as_os_str().is_empty(), "path must not be empty");
        let mut path = self.root.clone();
        for component in relative.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(name) => {
                    path.push(name);
                    ensure!(
                        !fs::symlink_metadata(&path)
                            .context("inspect path")?
                            .file_type()
                            .is_symlink(),
                        "symlinks are not supported"
                    );
                }
                _ => bail!("path must be repository-relative without parent traversal"),
            }
        }
        ensure!(
            !self.is_excluded(&path),
            "history cache is not source input"
        );
        let resolved = path.canonicalize().context("resolve path")?;
        ensure!(
            resolved.starts_with(&self.root),
            "path escapes repository root"
        );
        Ok(resolved)
    }
    pub fn execute(&self, name: &str, args: &Value) -> Value {
        match self.execute_inner(name, args) {
            Ok(value) => value,
            Err(error) => json!({"error": format!("{error:#}")}),
        }
    }
    fn execute_inner(&self, name: &str, args: &Value) -> Result<Value> {
        match name {
            "read_file" => self.read_file(
                string(args, "path")?,
                integer(args, "from_line")?,
                integer(args, "num_of_lines")?,
            ),
            "list_dir" => self.list_dir(string(args, "path")?),
            "string_search" => self.search(
                string(args, "path_wildcard")?,
                string(args, "text_wildcard")?,
            ),
            _ => bail!("unknown external tool {name}"),
        }
    }
    fn read_file(&self, path: &str, from: usize, count: usize) -> Result<Value> {
        ensure!(
            from >= 1 && count >= 1,
            "line arguments must be positive integers"
        );
        let text = read_text(&self.resolve(path)?)?;
        let all: Vec<_> = text.lines().collect();
        let available = all.len().saturating_sub(from - 1);
        let mut size = 0;
        let lines: Vec<_> = all
            .iter()
            .enumerate()
            .skip(from - 1)
            .take(count.min(MAX_ENTRIES))
            .take_while(|(_, line)| {
                size += line.len();
                size <= MAX_OUTPUT
            })
            .map(|(index, text)| json!({"line": index + 1, "text": text}))
            .collect();
        Ok(
            json!({"path": path, "from_line": from, "total_lines": all.len(), "truncated": lines.len() < available.min(count), "lines": lines}),
        )
    }
    fn list_dir(&self, path: &str) -> Result<Value> {
        let dir = self.resolve(path)?;
        ensure!(dir.is_dir(), "path is not a directory");
        // Bound discovery as well as model output. Very large directories fail rather
        // than returning a nondeterministic first slice of filesystem iteration order.
        let mut entries = Vec::new();
        for entry in fs::read_dir(dir).context("list directory")? {
            ensure!(
                entries.len() < MAX_VISITS,
                "directory exceeds {MAX_VISITS} entries"
            );
            let entry = entry?;
            if !self.is_excluded(&entry.path()) {
                entries.push(entry);
            }
        }
        entries.sort_by_key(|entry| entry.file_name());
        let total = entries.len();
        let mut output = Vec::new();
        let mut bytes = 0;
        for entry in entries.into_iter().take(MAX_ENTRIES) {
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF-8 file name"))?;
            bytes += name.len();
            if bytes > MAX_OUTPUT {
                break;
            }
            let kind = entry.file_type()?;
            output.push(json!({"name": name, "kind": if kind.is_symlink() { "symlink" } else if kind.is_dir() { "directory" } else if kind.is_file() { "file" } else { "other" }}));
        }
        Ok(json!({"path": path, "truncated": output.len() < total, "entries": output}))
    }
    fn search(&self, paths: &str, text: &str) -> Result<Value> {
        ensure!(
            !paths.is_empty() && paths.len() <= 512 && text.len() <= 512,
            "wildcard patterns must be at most 512 bytes and path pattern must not be empty"
        );
        ensure!(
            !paths.starts_with('/') && !paths.split('/').any(|s| s == ".."),
            "path wildcard must be repository-relative without parent traversal"
        );
        let paths = paths.strip_prefix("./").unwrap_or(paths);
        let mut matches = Vec::new();
        let mut skipped = 0;
        let mut bytes = 0;
        let mut truncated = false;
        // No symlinks, no .git internals. The caller must keep this tree stable.
        let walker = WalkDir::new(&self.root)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter()
            .filter_entry(|entry| {
                entry.depth() == 0
                    || (entry.file_name() != ".git" && !self.is_excluded(entry.path()))
            });
        for (visited, entry) in walker.enumerate() {
            if visited >= MAX_VISITS {
                truncated = true;
                break;
            }
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let relative = entry
                .path()
                .strip_prefix(&self.root)?
                .to_str()
                .context("non-UTF-8 file name")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            if !path_matches(paths, &relative) {
                continue;
            }
            let contents = match self.resolve(&relative).and_then(|path| read_text(&path)) {
                Ok(text) => text,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            for (index, line) in contents.lines().enumerate() {
                if wildcard(text, line) {
                    bytes += relative.len() + line.len();
                    if matches.len() >= MAX_ENTRIES || bytes > MAX_OUTPUT {
                        truncated = true;
                        break;
                    }
                    matches.push(json!({"path": relative, "line": index + 1, "text": line}));
                }
            }
            if truncated {
                break;
            }
        }
        Ok(
            json!({"matches": matches, "truncated": truncated, "skipped_files_or_errors": skipped, "excluded_directories": [".git"], "followed_symlinks": false}),
        )
    }
}
fn string<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("{key} must be a string"))
}
fn integer(args: &Value, key: &str) -> Result<usize> {
    args.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .with_context(|| format!("{key} must be a nonnegative integer"))
}
fn read_text(path: &Path) -> Result<String> {
    ensure!(path.is_file(), "path is not a regular file");
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take((MAX_FILE + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= MAX_FILE,
        "file exceeds 1 MiB inspection limit"
    );
    let text = String::from_utf8(bytes).context("file is not UTF-8")?;
    ensure!(!text.contains('\0'), "binary file is not supported");
    Ok(text)
}

// Only * and ? have special meaning; brackets, braces, and backslashes are literal.
pub fn wildcard(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut i, mut j, mut star, mut retry) = (0, 0, None, 0);
    while j < t.len() {
        if i < p.len() && (p[i] == '?' || p[i] == t[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some(i);
            i += 1;
            retry = j;
        } else if let Some(s) = star {
            retry += 1;
            j = retry;
            i = s + 1;
        } else {
            return false;
        }
    }
    while i < p.len() && p[i] == '*' {
        i += 1;
    }
    i == p.len()
}
pub fn path_matches(pattern: &str, text: &str) -> bool {
    let p: Vec<_> = pattern.split('/').collect();
    let t: Vec<_> = text.split('/').collect();
    let mut row = vec![false; t.len() + 1];
    row[0] = true;
    for segment in p {
        let mut next = vec![false; row.len()];
        if segment == "**" {
            next[0] = row[0];
        }
        for j in 1..row.len() {
            next[j] = if segment == "**" {
                row[j] || next[j - 1]
            } else {
                row[j - 1] && wildcard(segment, t[j - 1])
            };
        }
        row = next;
    }
    row[t.len()]
}
