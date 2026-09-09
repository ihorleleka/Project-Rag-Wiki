const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

// Single source of truth for the shared release version. Both kb-service and
// wiki-kit ride the same Git tag (see the repository README "Versioning"), so
// the image tag, current service version, and reported wiki-kit version all
// derive from this constant. Schema 7 / tool contract 5 was a hard break from
// the schema 6 / contract 4 line, so only the current release is supported;
// older images classify as incompatible rather than merely outdated.
const CURRENT_SERVICE_VERSION = "0.1.12";
const IMAGE_REPOSITORY = "ihorleleka/project-rag-wiki";
const DEFAULT_IMAGE = `${IMAGE_REPOSITORY}:${CURRENT_SERVICE_VERSION}`;

const SERVICE_COMPATIBILITY = Object.freeze({
  [CURRENT_SERVICE_VERSION]: Object.freeze({
    indexSchemaVersion: 7,
    minimumMcpToolContractVersion: 5,
    maximumMcpToolContractVersion: 5,
    requiredTools: Object.freeze([
      "wiki_search",
      "wiki_read",
      "wiki_list",
      "wiki_schema_report",
      "wiki_tree",
      "wiki_write",
      "wiki_capture",
      "wiki_delete",
      "wiki_rename",
    ]),
  }),
});

const CURRENT_SERVICE = SERVICE_COMPATIBILITY[CURRENT_SERVICE_VERSION];
const COMPATIBILITY = Object.freeze({
  wikiKitVersion: CURRENT_SERVICE_VERSION,
  currentServiceVersion: CURRENT_SERVICE_VERSION,
  minimumServiceVersion: CURRENT_SERVICE_VERSION,
  maximumServiceVersion: CURRENT_SERVICE_VERSION,
  indexSchemaVersion: CURRENT_SERVICE.indexSchemaVersion,
  minimumMcpToolContractVersion: CURRENT_SERVICE.minimumMcpToolContractVersion,
  maximumMcpToolContractVersion: CURRENT_SERVICE.maximumMcpToolContractVersion,
  requiredTools: CURRENT_SERVICE.requiredTools,
  services: SERVICE_COMPATIBILITY,
});

function normalizeContainerName(input) {
  return input
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 32) || "kb";
}

function canonicalRepositoryRoot(repositoryRoot) {
  const resolved = path.resolve(repositoryRoot);
  let canonical = resolved;
  try {
    canonical = fs.realpathSync.native(resolved);
  } catch {
    // The resolved absolute path is still deterministic if realpath is unavailable.
  }
  const portable = canonical.replace(/\\/g, "/");
  return process.platform === "win32" ? portable.toLowerCase() : portable;
}

function deriveResourceNames(repositoryRoot, env = process.env) {
  const canonicalRoot = canonicalRepositoryRoot(repositoryRoot);
  const rootHash = crypto
    .createHash("sha256")
    .update(canonicalRoot, "utf8")
    .digest("hex")
    .slice(0, 12);
  const repoName = normalizeContainerName(path.basename(canonicalRoot) || "repo").slice(0, 16);
  const repositoryIdentity = `${repoName}-${rootHash}`;

  return {
    kbVolume: env.KB_VOLUME || `${repositoryIdentity}-kb-data`,
    hfCacheVolume: env.HF_CACHE_VOLUME || "hf-cache",
    containerName: normalizeContainerName(
      env.KB_CONTAINER_NAME || `${repositoryIdentity}-kb`
    ),
  };
}

function parseVersion(input) {
  const match = String(input || "").match(/^(\d+)\.(\d+)\.(\d+)$/);
  return match ? match.slice(1).map(Number) : null;
}

function classifyServiceVersion(serviceVersion) {
  if (!parseVersion(serviceVersion) || !SERVICE_COMPATIBILITY[serviceVersion]) {
    return "incompatible";
  }
  return serviceVersion === COMPATIBILITY.currentServiceVersion ? "current" : "outdated";
}

function classifyCompatibility(metadata) {
  const serviceState = classifyServiceVersion(metadata.serviceVersion);
  if (serviceState === "incompatible") return serviceState;
  const expected = SERVICE_COMPATIBILITY[metadata.serviceVersion];
  const schema = metadata.indexSchemaVersion;
  const toolContract = metadata.mcpToolContractVersion;
  if (schema === undefined || Number(schema) !== expected.indexSchemaVersion) {
    return "incompatible";
  }
  if (
    toolContract === undefined ||
    Number(toolContract) < expected.minimumMcpToolContractVersion ||
    Number(toolContract) > expected.maximumMcpToolContractVersion
  ) {
    return "incompatible";
  }
  return serviceState;
}

module.exports = {
  COMPATIBILITY,
  DEFAULT_IMAGE,
  SERVICE_COMPATIBILITY,
  canonicalRepositoryRoot,
  classifyCompatibility,
  classifyServiceVersion,
  deriveResourceNames,
};

