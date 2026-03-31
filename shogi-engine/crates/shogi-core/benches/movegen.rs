use criterion::{criterion_group, criterion_main, Criterion};
use shogi_core::{GameState, MoveList};

fn bench_legal_moves_opening(c: &mut Criterion) {
    c.bench_function("legal_moves_opening", |b| {
        b.iter(|| {
            let mut game = GameState::new();
            game.legal_moves()
        });
    });
}

fn bench_legal_moves_opening_hot_path(c: &mut Criterion) {
    c.bench_function("legal_moves_opening_hot_path", |b| {
        let mut move_list = MoveList::new();
        b.iter(|| {
            let mut game = GameState::new();
            game.generate_legal_moves_into(&mut move_list);
        });
    });
}

fn bench_make_unmake(c: &mut Criterion) {
    c.bench_function("make_unmake_cycle", |b| {
        let mut game = GameState::new();
        let moves = game.legal_moves();
        b.iter(|| {
            for &mv in &moves {
                let undo = game.make_move(mv);
                game.unmake_move(mv, undo);
            }
        });
    });
}

fn bench_attack_map_from_scratch(c: &mut Criterion) {
    use shogi_core::Position;
    c.bench_function("attack_map_from_scratch", |b| {
        let pos = Position::startpos();
        b.iter(|| {
            shogi_core::attack::compute_attack_map(&pos)
        });
    });
}

criterion_group!(
    benches,
    bench_legal_moves_opening,
    bench_legal_moves_opening_hot_path,
    bench_make_unmake,
    bench_attack_map_from_scratch
);
criterion_main!(benches);
