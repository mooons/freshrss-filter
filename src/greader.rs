use crate::config::FreshRssConfig;
use anyhow::{Result, anyhow};
use reqwest::{Client, Url};

#[derive(Clone)]
pub struct GReaderClient {
    client: Client,
    base: Url,
    auth: GReaderAuth,
}

#[derive(Clone)]
enum GReaderAuth {
    Basic { username: String, password: String },
    GoogleLogin { authorization: String },
}

pub fn has_auth_config(cfg: &FreshRssConfig) -> bool {
    cfg.greader_googlelogin_auth
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
        || (cfg
            .greader_username
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
            && cfg
                .greader_password
                .as_deref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false))
}

pub fn build_client(cfg: &FreshRssConfig) -> Result<GReaderClient> {
    let base = Url::parse(&cfg.base_url)?;
    let client = Client::builder().user_agent(&cfg.user_agent).build()?;
    let auth = if let Some(token) = cfg
        .greader_googlelogin_auth
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let lower = token.to_ascii_lowercase();
        let authorization = if lower.starts_with("googlelogin auth=") {
            token.to_string()
        } else {
            format!("GoogleLogin auth={}", token)
        };
        GReaderAuth::GoogleLogin { authorization }
    } else if let (Some(username), Some(password)) = (
        cfg.greader_username.as_deref().map(str::trim),
        cfg.greader_password.as_deref().map(str::trim),
    ) {
        if username.is_empty() || password.is_empty() {
            return Err(anyhow!(
                "greader_credentials_missing: set greader_googlelogin_auth or both greader_username and greader_password"
            ));
        }
        GReaderAuth::Basic {
            username: username.to_string(),
            password: password.to_string(),
        }
    } else {
        return Err(anyhow!(
            "greader_credentials_missing: set greader_googlelogin_auth or both greader_username and greader_password"
        ));
    };

    Ok(GReaderClient {
        client,
        base,
        auth,
    })
}

impl GReaderClient {
    pub async fn add_label(&self, item_id: i64, label: &str) -> Result<()> {
        let url = self.base.join("/api/greader.php/reader/api/0/edit-tag")?;
        let tag = format!("user/-/label/{}", label);
        let req = self
            .client
            .post(url)
            .form(&[("i", item_id.to_string()), ("a", tag)]);
        let req = match &self.auth {
            GReaderAuth::Basic { username, password } => req.basic_auth(username, Some(password)),
            GReaderAuth::GoogleLogin { authorization } => {
                req.header("Authorization", authorization)
            }
        };
        let resp = req.send().await?;
        if !resp.status().is_success() {
            return Err(anyhow!("greader_edit_tag_error: {}", resp.status()));
        }
        Ok(())
    }
}
