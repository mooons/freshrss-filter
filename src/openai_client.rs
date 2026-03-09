use crate::config::OpenAiConfig;
use anyhow::{Result, anyhow};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, instrument};

use std::fmt;

#[derive(Debug, Clone)]
pub struct OpenAiApiError {
    pub status: StatusCode,
    pub body: Value,
}

impl OpenAiApiError {
    pub fn new(status: StatusCode, body: Value) -> Self {
        Self { status, body }
    }
}

impl fmt::Display for OpenAiApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "openai_error: status={} body={}", self.status, self.body)
    }
}

impl std::error::Error for OpenAiApiError {}

#[derive(Clone)]
pub struct OpenAiClient {
    client: Client,
    cfg: OpenAiConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ClassifierResponse {
    pub is_ad: bool,
    pub confidence: f32,
    pub reason: String,
    pub is_worth: Option<bool>,
    pub worth_confidence: Option<f32>,
    pub worth_reason: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct AdSignal {
    pub is_ad: bool,
    pub confidence: f32,
    pub reason: String,
}

#[derive(Debug, Deserialize, Clone)]
struct WorthSignal {
    pub is_worth: bool,
    pub confidence: f32,
    pub reason: String,
}

impl OpenAiClient {
    pub fn new(cfg: OpenAiConfig) -> Self {
        let client = Client::builder().build().unwrap();
        Self { client, cfg }
    }

    #[instrument(name = "Reviewing content", skip(self, text))]
    pub async fn classify(&self, text: &str) -> Result<ClassifierResponse> {
        #[derive(Serialize)]
        struct ReqBody<'a> {
            model: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            response_format: Option<RespFmt>,
            messages: Vec<Message<'a>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            #[serde(skip_serializing_if = "Option::is_none")]
            max_tokens: Option<u32>,
        }
        #[derive(Serialize)]
        struct RespFmt {
            r#type: &'static str,
        }
        #[derive(Serialize)]
        struct Message<'a> {
            role: &'static str,
            content: &'a str,
        }

        let use_json_object_response_format = !self.cfg.api_base.contains("anthropic.com");

        let body = ReqBody {
            model: &self.cfg.model,
            response_format: use_json_object_response_format.then_some(RespFmt {
                r#type: "json_object",
            }),
            messages: vec![
                Message {
                    role: "system",
                    content: &self.cfg.system_prompt,
                },
                Message {
                    role: "user",
                    content: text,
                },
            ],
            temperature: self.cfg.temperature,
            max_tokens: self.cfg.max_tokens,
        };

        let url = format!(
            "{}/chat/completions",
            self.cfg.api_base.trim_end_matches('/')
        );
        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let v: Value = resp.json().await?;

        if let Some(err) = v.get("error") {
            return Err(OpenAiApiError::new(status, err.clone()).into());
        }

        if !status.is_success() {
            return Err(OpenAiApiError::new(status, v).into());
        }

        // Extract content
        let raw = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}");
        let content = strip_code_fences(raw);
        parse_classifier_response(&content, raw)
    }
}

fn parse_classifier_response(content: &str, raw: &str) -> Result<ClassifierResponse> {
    match serde_json::from_str::<AdSignal>(content) {
        Ok(parsed) => Ok(ClassifierResponse {
            is_ad: parsed.is_ad,
            confidence: parsed.confidence,
            reason: parsed.reason,
            is_worth: None,
            worth_confidence: None,
            worth_reason: None,
        }),
        Err(primary_err) => {
            let v: Value = serde_json::from_str(content).map_err(|secondary_err| {
                anyhow!(
                    "parse_classifier_response_failed: {} (array_parse_error: {}) raw={}",
                    primary_err,
                    secondary_err,
                    raw
                )
            })?;

            if let Some(obj) = v.as_object() {
                if let Ok(worth) = serde_json::from_value::<WorthSignal>(Value::Object(obj.clone())) {
                    return Ok(ClassifierResponse {
                        is_ad: false,
                        confidence: worth.confidence,
                        reason: worth.reason.clone(),
                        is_worth: Some(worth.is_worth),
                        worth_confidence: Some(worth.confidence),
                        worth_reason: Some(worth.reason),
                    });
                }
                return Err(anyhow!(
                    "parse_classifier_response_failed: expected ad/worth object or array raw={}",
                    raw
                ));
            }

            let arr = v.as_array().ok_or_else(|| {
                anyhow!(
                    "parse_classifier_response_failed: expected ad/worth object or array raw={}",
                    raw
                )
            })?;

            let ad_signals: Vec<AdSignal> = arr
                .iter()
                .filter_map(|item| serde_json::from_value::<AdSignal>(item.clone()).ok())
                .collect();
            let worth_signals: Vec<WorthSignal> = arr
                .iter()
                .filter_map(|item| serde_json::from_value::<WorthSignal>(item.clone()).ok())
                .collect();

            if ad_signals.is_empty() && worth_signals.is_empty() {
                return Err(anyhow!(
                    "parse_classifier_response_failed: no valid ad/worth objects in array raw={}",
                    raw
                ));
            }

            debug!(
                error = %primary_err,
                total_count = arr.len(),
                ad_count = ad_signals.len(),
                worth_count = worth_signals.len(),
                "classifier_response_array_detected"
            );

            let best_ad_true = ad_signals
                .iter()
                .filter(|r| r.is_ad)
                .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                .cloned();
            let best_ad_any = ad_signals
                .iter()
                .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                .cloned();
            let best_worth_false = worth_signals
                .iter()
                .filter(|r| !r.is_worth)
                .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                .cloned();
            let best_worth_any = worth_signals
                .iter()
                .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
                .cloned();

            let ad_selected = best_ad_true.or(best_ad_any.clone());
            let worth_selected = best_worth_false.or(best_worth_any);

            if ad_selected.is_none() && worth_selected.is_none() {
                return Err(anyhow!(
                    "parse_classifier_response_failed: empty array raw={}",
                    raw
                ));
            }

            Ok(ClassifierResponse {
                is_ad: ad_selected.as_ref().map(|a| a.is_ad).unwrap_or(false),
                confidence: ad_selected.as_ref().map(|a| a.confidence).unwrap_or_else(|| {
                    worth_selected
                        .as_ref()
                        .map(|w| w.confidence)
                        .unwrap_or(0.0)
                }),
                reason: ad_selected
                    .as_ref()
                    .map(|a| a.reason.clone())
                    .or_else(|| worth_selected.as_ref().map(|w| w.reason.clone()))
                    .unwrap_or_else(|| "no_reason".to_string()),
                is_worth: worth_selected.as_ref().map(|w| w.is_worth),
                worth_confidence: worth_selected.as_ref().map(|w| w.confidence),
                worth_reason: worth_selected.as_ref().map(|w| w.reason.clone()),
            })
        }
    }
}

fn strip_code_fences(s: &str) -> String {
    let t = s.trim();
    if t.starts_with("```") {
        // remove first line fence and trailing fence
        let mut lines = t.lines();
        let _first = lines.next();
        let rest: String = lines.collect::<Vec<_>>().join("\n");
        let trimmed = rest.trim_end();
        if trimmed.ends_with("```") {
            return trimmed.trim_end_matches("```").trim().to_string();
        }
        return trimmed.to_string();
    }
    t.to_string()
}
