-- EBRCS Streaming Database Schema Migration
-- Version: 001
-- Description: Initial schema for user authentication, products, sessions, and orders

-- Enable UUID extension (PostgreSQL)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. USERS TABLE
-- =============================================================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- bcrypt hashed
    role VARCHAR(10) NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    email VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);

COMMENT ON TABLE users IS 'User accounts with role-based access (user/admin)';
COMMENT ON COLUMN users.role IS 'user: checkout only, admin: can add products';

-- =============================================================================
-- 2. PRODUCTS TABLE
-- =============================================================================
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) CHECK (price >= 0),
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX idx_products_name ON products(name);
CREATE INDEX idx_products_is_active ON products(is_active);
CREATE INDEX idx_products_created_by ON products(created_by);

COMMENT ON TABLE products IS 'Product catalog managed by admin users';
COMMENT ON COLUMN products.price IS 'Optional: for future payment integration';

-- =============================================================================
-- 3. PRODUCT IMAGES TABLE
-- =============================================================================
CREATE TABLE product_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    image_path VARCHAR(500) NOT NULL,
    embedding_id INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_product_images_product_id ON product_images(product_id);
CREATE INDEX idx_product_images_embedding_id ON product_images(embedding_id);
CREATE UNIQUE INDEX idx_product_images_embedding_id_unique ON product_images(embedding_id);

COMMENT ON TABLE product_images IS 'Maps product images to embedding array indices';
COMMENT ON COLUMN product_images.image_path IS 'Local path or S3 URL to product image';
COMMENT ON COLUMN product_images.embedding_id IS 'Index in embeddings.npy file (0-based)';

-- =============================================================================
-- 4. CHECKOUT SESSIONS TABLE
-- =============================================================================
CREATE TABLE checkout_sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_active TIMESTAMP NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'expired')),
    metadata JSONB
);

CREATE INDEX idx_checkout_sessions_user_id ON checkout_sessions(user_id);
CREATE INDEX idx_checkout_sessions_status ON checkout_sessions(status);
CREATE INDEX idx_checkout_sessions_last_active ON checkout_sessions(last_active);

COMMENT ON TABLE checkout_sessions IS 'Tracks active and historical checkout sessions';
COMMENT ON COLUMN checkout_sessions.id IS 'UUID from SessionManager, matches in-memory session';
COMMENT ON COLUMN checkout_sessions.metadata IS 'Optional: store ROI config, video upload info, etc.';

-- =============================================================================
-- 5. BILLING ITEMS TABLE
-- =============================================================================
CREATE TABLE billing_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES checkout_sessions(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    quantity INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    avg_score DECIMAL(5, 4) CHECK (avg_score >= 0 AND avg_score <= 1),
    added_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP
);

CREATE INDEX idx_billing_items_session_id ON billing_items(session_id);
CREATE INDEX idx_billing_items_product_id ON billing_items(product_id);
CREATE UNIQUE INDEX idx_billing_items_session_product ON billing_items(session_id, product_id);

COMMENT ON TABLE billing_items IS 'Real-time shopping cart for active checkout sessions';
COMMENT ON COLUMN billing_items.avg_score IS 'Average FAISS similarity score for this product in session';

-- =============================================================================
-- 6. ORDERS TABLE
-- =============================================================================
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    session_id UUID REFERENCES checkout_sessions(id) ON DELETE SET NULL,
    total_items INTEGER NOT NULL CHECK (total_items > 0),
    total_amount DECIMAL(10, 2) CHECK (total_amount >= 0),
    confirmed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    payment_status VARCHAR(20) NOT NULL DEFAULT 'mock_paid' CHECK (payment_status IN ('mock_paid', 'pending', 'completed', 'failed'))
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_session_id ON orders(session_id);
CREATE INDEX idx_orders_confirmed_at ON orders(confirmed_at DESC);

COMMENT ON TABLE orders IS 'Confirmed purchase receipts (immutable after creation)';
COMMENT ON COLUMN orders.payment_status IS 'mock_paid: frontend simulation, completed: real payment';

-- =============================================================================
-- 7. ORDER ITEMS TABLE
-- =============================================================================
CREATE TABLE order_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    avg_score DECIMAL(5, 4),
    unit_price DECIMAL(10, 2) CHECK (unit_price >= 0)
);

CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

COMMENT ON TABLE order_items IS 'Line items for each order (snapshot of billing_items at confirmation)';

-- =============================================================================
-- SAMPLE DATA (OPTIONAL)
-- =============================================================================

-- Create default admin user (password: 'admin123' - CHANGE IN PRODUCTION!)
-- bcrypt hash of 'admin123': $2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqJqJ8qToy
INSERT INTO users (username, password_hash, role, email)
VALUES ('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqJqJ8qToy', 'admin', 'admin@example.com');

-- Create sample regular user (password: 'user123')
-- bcrypt hash of 'user123': $2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW
INSERT INTO users (username, password_hash, role, email)
VALUES ('user1', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'user', 'user1@example.com');

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for products table
CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for billing_items table
CREATE TRIGGER update_billing_items_updated_at BEFORE UPDATE ON billing_items
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- VIEWS (OPTIONAL)
-- =============================================================================

-- View: Active sessions with user info
CREATE VIEW v_active_sessions AS
SELECT
    cs.id AS session_id,
    cs.user_id,
    u.username,
    cs.created_at,
    cs.last_active,
    COUNT(bi.id) AS item_count,
    SUM(bi.quantity) AS total_quantity
FROM checkout_sessions cs
LEFT JOIN users u ON cs.user_id = u.id
LEFT JOIN billing_items bi ON cs.id = bi.session_id
WHERE cs.status = 'active'
GROUP BY cs.id, cs.user_id, u.username, cs.created_at, cs.last_active;

-- View: Order history with details
CREATE VIEW v_order_history AS
SELECT
    o.id AS order_id,
    o.user_id,
    u.username,
    o.confirmed_at,
    o.total_items,
    o.total_amount,
    o.payment_status,
    COUNT(oi.id) AS line_items_count,
    COALESCE(SUM(oi.quantity), 0) AS total_quantity
FROM orders o
LEFT JOIN users u ON o.user_id = u.id
LEFT JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.id, o.user_id, u.username, o.confirmed_at, o.total_items, o.total_amount, o.payment_status;

-- =============================================================================
-- CLEANUP FUNCTION (For expired sessions)
-- =============================================================================

CREATE OR REPLACE FUNCTION cleanup_expired_sessions(expire_after_hours INTEGER DEFAULT 24)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM checkout_sessions
    WHERE status = 'active'
      AND last_active < NOW() - (expire_after_hours || ' hours')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_expired_sessions IS 'Delete inactive checkout sessions older than N hours';

-- Usage example:
-- SELECT cleanup_expired_sessions(24);  -- Clean sessions inactive for 24+ hours
