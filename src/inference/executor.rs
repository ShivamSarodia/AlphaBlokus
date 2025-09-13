use crate::inference;

pub trait Executor: Send + Sync + 'static {
    fn execute(&self, requests: Vec<inference::Request>) -> Vec<inference::Response>;
}
