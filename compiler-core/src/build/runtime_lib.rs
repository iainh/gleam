use camino::Utf8PathBuf;

use crate::Error;

use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub struct RuntimeArtifacts {
    pub runtime_lib: Utf8PathBuf,
    pub additional_libs: Vec<Utf8PathBuf>,
}

pub fn locate_runtime_artifacts() -> Result<RuntimeArtifacts, Error> {
    if let Some(path) = env::var_os("GLEAM_RUNTIME_LIB") {
        let path = PathBuf::from(path);
        return runtime_artifacts_from_lib(path).ok_or_else(|| missing_runtime_error(vec![]));
    }

    let mut searched = Vec::new();

    if let Some(dir) = env::var_os("GLEAM_RUNTIME_LIB_DIR") {
        let dir = PathBuf::from(dir);
        searched.push(dir.clone());
        if let Some(artifacts) = runtime_artifacts_from_dir(&dir) {
            return Ok(artifacts);
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidates = candidate_directories(exe_dir);
            for dir in candidates {
                searched.push(dir.clone());
                if let Some(artifacts) = runtime_artifacts_from_dir(&dir) {
                    return Ok(artifacts);
                }
            }
        }
    }

    Err(missing_runtime_error(searched))
}

fn runtime_artifacts_from_lib(path: PathBuf) -> Option<RuntimeArtifacts> {
    if !path.exists() {
        return None;
    }
    let dir = path.parent()?.to_path_buf();
    let runtime_lib = Utf8PathBuf::from_path_buf(path).ok()?;
    let (mut additional_libs, has_gc) = collect_supporting_libs(&dir);
    if !has_gc {
        return None;
    }
    additional_libs.sort();
    Some(RuntimeArtifacts {
        runtime_lib,
        additional_libs,
    })
}

fn runtime_artifacts_from_dir(dir: &Path) -> Option<RuntimeArtifacts> {
    let runtime = dir.join("libruntime_cranelift.a");
    runtime_artifacts_from_lib(runtime)
}

fn collect_supporting_libs(dir: &Path) -> (Vec<Utf8PathBuf>, bool) {
    let mut libs = Vec::new();
    let mut has_gc = false;

    if let Some(gc) = find_library(dir, "libgc.a") {
        has_gc = true;
        libs.push(gc);
    }

    if let Some(cord) = find_library(dir, "libcord.a") {
        libs.push(cord);
    }

    (libs, has_gc)
}

fn find_library(dir: &Path, name: &str) -> Option<Utf8PathBuf> {
    let direct = dir.join(name);
    if direct.exists() {
        return Utf8PathBuf::from_path_buf(direct).ok();
    }

    let mut current = Some(dir.to_path_buf());
    while let Some(path) = current {
        let build_dir = path.join("build");
        if let Some(found) = find_in_build_dir(&build_dir, name) {
            return Some(found);
        }
        current = path.parent().map(Path::to_path_buf);
    }

    None
}

fn find_in_build_dir(build_dir: &Path, name: &str) -> Option<Utf8PathBuf> {
    if !build_dir.is_dir() {
        return None;
    }

    let entries = fs::read_dir(build_dir).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("out").join("lib").join(name);
        if candidate.exists() {
            if let Ok(path) = Utf8PathBuf::from_path_buf(candidate) {
                return Some(path);
            }
        }
    }
    None
}

fn candidate_directories(exe_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    dirs.push(exe_dir.to_path_buf());
    dirs.push(exe_dir.join("deps"));
    dirs.push(exe_dir.join("lib"));

    if let Some(parent) = exe_dir.parent() {
        dirs.push(parent.join("release"));
        dirs.push(parent.join("debug"));
        dirs.push(parent.join("lib"));
        dirs.push(parent.join("deps"));
    }

    dirs.into_iter().filter(|d| d.exists()).collect()
}

fn missing_runtime_error(searched: Vec<PathBuf>) -> Error {
    let searched: Vec<String> = searched
        .into_iter()
        .map(|p| p.display().to_string())
        .collect();
    Error::CraneliftCodegen {
        message: format!(
            "unable to locate Cranelift runtime static library. Set the \"GLEAM_RUNTIME_LIB\" \
             environment variable to the path of libruntime_cranelift.a or \"GLEAM_RUNTIME_LIB_DIR\" \
             to a directory containing the runtime libraries. Searched directories: {}",
            if searched.is_empty() {
                "<none>".into()
            } else {
                searched.join(", ")
            }
        ),
    }
}
