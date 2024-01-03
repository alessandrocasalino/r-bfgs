use std::collections::VecDeque;
use crate::settings::Settings;

pub(super) struct HistoryPoint {
    pub(super) s : Vec<f64>,
    pub(super) y : Vec<f64>
}

pub(super) fn fifo_operation (q : &mut VecDeque<HistoryPoint>, s : Vec<f64>, y : Vec<f64>, settings: &Settings) {
    q.push_front(HistoryPoint{s, y});
    if q.len() > settings.history_depth {
        q.pop_back();
    }
}
