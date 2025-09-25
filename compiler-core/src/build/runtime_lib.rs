use camino::{Utf8Path, Utf8PathBuf};

use crate::{Error, io::FileSystemReader};

use std::{env, path::PathBuf};

#[derive(Debug)]
pub struct RuntimeArtifacts {
    pub runtime_lib: Utf8PathBuf,
    pub additional_libs: Vec<Utf8PathBuf>,
}

pub fn locate_runtime_artifacts(io: &impl FileSystemReader) -> Result<RuntimeArtifacts, Error> {
    if let Some(path) = env::var_os("GLEAM_RUNTIME_LIB") {
        let path = PathBuf::from(path);
        if let Ok(path) = Utf8PathBuf::from_path_buf(path) {
            if let Some(artifacts) = runtime_artifacts_from_lib(io, &path) {
                return Ok(artifacts);
            } else {
                return Err(missing_runtime_error(vec![path]));
            }
        } else {
            return Err(missing_runtime_error(vec![]));
        }
    }

    let mut searched: Vec<Utf8PathBuf> = Vec::new();

    if let Some(dir) = env::var_os("GLEAM_RUNTIME_LIB_DIR") {
        let dir = PathBuf::from(dir);
        if let Ok(dir) = Utf8PathBuf::from_path_buf(dir) {
            searched.push(dir.clone());
            if let Some(artifacts) = runtime_artifacts_from_dir(io, &dir) {
                return Ok(artifacts);
            }
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Ok(exe) = Utf8PathBuf::from_path_buf(exe) {
            if let Some(exe_dir) = exe.parent() {
                for dir in candidate_directories(io, exe_dir) {
                    searched.push(dir.clone());
                    if let Some(artifacts) = runtime_artifacts_from_dir(io, &dir) {
                        return Ok(artifacts);
                    }
                }
            }
        }
    }

    Err(missing_runtime_error(searched))
}

fn runtime_artifacts_from_lib(
    io: &impl FileSystemReader,
    path: &Utf8Path,
) -> Option<RuntimeArtifacts> {
    if !io.is_file(path) {
        return None;
    }
    let dir = path.parent()?.to_path_buf();
    let runtime_lib = path.to_path_buf();
    let (mut additional_libs, has_gc) = collect_supporting_libs(io, dir.as_path());
    if !has_gc {
        return None;
    }
    additional_libs.sort();
    Some(RuntimeArtifacts {
        runtime_lib,
        additional_libs,
    })
}

fn runtime_artifacts_from_dir(
    io: &impl FileSystemReader,
    dir: &Utf8Path,
) -> Option<RuntimeArtifacts> {
    let runtime = dir.join("libruntime_cranelift.a");
    runtime_artifacts_from_lib(io, &runtime)
}

fn collect_supporting_libs(io: &impl FileSystemReader, dir: &Utf8Path) -> (Vec<Utf8PathBuf>, bool) {
    let mut libs = Vec::new();
    let mut has_gc = false;

    if let Some(gc) = find_library(io, dir, "libgc.a") {
        has_gc = true;
        libs.push(gc);
    }

    if let Some(cord) = find_library(io, dir, "libcord.a") {
        libs.push(cord);
    }

    (libs, has_gc)
}

fn find_library(io: &impl FileSystemReader, dir: &Utf8Path, name: &str) -> Option<Utf8PathBuf> {
    let direct = dir.join(name);
    if io.is_file(&direct) {
        return Some(direct);
    }

    let mut current = Some(dir.to_path_buf());
    while let Some(path) = current {
        let build_dir = path.join("build");
        if let Some(found) = find_in_build_dir(io, &build_dir, name) {
            return Some(found);
        }
        current = path.parent().map(|parent| parent.to_path_buf());
    }

    None
}

fn find_in_build_dir(
    io: &impl FileSystemReader,
    build_dir: &Utf8Path,
    name: &str,
) -> Option<Utf8PathBuf> {
    if !io.is_directory(build_dir) {
        return None;
    }

    let entries = io.read_dir(build_dir).ok()?;
    for entry in entries {
        let entry = entry.ok()?;
        let candidate = entry.into_path().join("out").join("lib").join(name);
        if io.is_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn candidate_directories(io: &impl FileSystemReader, exe_dir: &Utf8Path) -> Vec<Utf8PathBuf> {
    let mut dirs = Vec::new();
    dirs.push(exe_dir.to_path_buf());
    dirs.push(exe_dir.join("deps"));
    dirs.push(exe_dir.join("lib"));
    dirs.push(exe_dir.join("gleam"));

    if let Some(parent) = exe_dir.parent() {
        dirs.push(parent.join("release"));
        dirs.push(parent.join("debug"));
        dirs.push(parent.join("lib"));
        dirs.push(parent.join("deps"));
        dirs.push(parent.join("lib/gleam"));
    }

    dirs.into_iter()
        .filter(|dir| io.is_directory(dir))
        .collect()
}

fn missing_runtime_error(searched: Vec<Utf8PathBuf>) -> Error {
    let searched: Vec<String> = searched.into_iter().map(|p| p.to_string()).collect();
    Error::NativeCodegen {
        message: format!(
            "unable to locate native (Cranelift) runtime static library. Set the \"GLEAM_RUNTIME_LIB\" \
             environment variable to the path of libruntime_cranelift.a or \"GLEAM_RUNTIME_LIB_DIR\" \
             to a directory containing the runtime libraries (run `make install` to place them \
             automatically). Searched directories: {}",
            if searched.is_empty() {
                "<none>".into()
            } else {
                searched.join(", ")
            }
        ),
    }
}
