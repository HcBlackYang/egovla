# utils/scheduler_utils.py
import yaml

class AdaptiveLossWeightScheduler:
    """
    自适应损失权重调度器。
    根据验证集指标（存储在报告中）来动态调整不同损失项的权重。
    """
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.initial_weights = self.config['loss_weights']
        self.current_weights = self.initial_weights.copy()
        
    def update_weights(self, validation_report: dict, current_epoch: int):
        """
        根据验证报告更新权重。
        这是一个简单的示例逻辑，可以根据需求定制。
        """
        print(f"Epoch {current_epoch}: 更新损失权重...")
        
        # 示例逻辑：如果语义蒸馏的验证损失停止下降，就降低其权重
        if "semantic_val_loss" in validation_report:
            if validation_report["semantic_val_loss"] > validation_report.get("prev_semantic_val_loss", float('inf')):
                self.current_weights['semantic'] *= 0.98
                print(f"  - 语义验证损失未改善，权重降低至 {self.current_weights['semantic']:.4f}")

        # 示例逻辑：在训练后期，增加对抗损失的权重
        if current_epoch > self.config.get('adversarial_start_epoch', 50):
            if 'adversarial' in self.current_weights:
                self.current_weights['adversarial'] *= 1.02
                print(f"  - 训练后期，对抗损失权重增加至 {self.current_weights['adversarial']:.4f}")

        return self.current_weights

    def get_weights(self):
        return self.current_weights